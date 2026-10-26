###############################################################################
# FeedbackIQ -- AWS Fargate deployment.
#
# The point of putting this in Terraform rather than clicking through the
# console: the LLM API key, the GitHub token, and the Slack webhook are the
# three things that must never live in an environment variable in a task
# definition. Here they are Secrets Manager ARNs injected at container start,
# and the IAM policy that reads them is scoped to exactly those three ARNs.
#
#   terraform init
#   terraform plan  -var="image_digest=sha256:..." -var="environment=prod"
#   terraform apply -var="image_digest=sha256:..." -var="environment=prod"
#
# State belongs in S3 with DynamoDB locking -- see the commented backend block.
###############################################################################

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # backend "s3" {
  #   bucket         = "acme-tfstate"
  #   key            = "feedbackiq/terraform.tfstate"
  #   region         = "us-east-2"
  #   dynamodb_table = "tfstate-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Application = "feedbackiq"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

variable "region" { default = "us-east-2" }
variable "environment" { default = "prod" }
variable "image_digest" {
  type        = string
  description = "Image digest from CI. Digest, not tag: the deploy is immutable."
}
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "public_subnet_ids" { type = list(string) }
variable "desired_count" { default = 2 }

locals {
  name  = "feedbackiq-${var.environment}"
  image = "${aws_ecr_repository.app.repository_url}@${var.image_digest}"
}

###############################################################################
# Registry
###############################################################################
resource "aws_ecr_repository" "app" {
  name                 = "feedbackiq"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

###############################################################################
# Secrets -- values are set out of band (CLI or console), never in Terraform,
# so they never land in state.
###############################################################################
resource "aws_secretsmanager_secret" "anthropic" { name = "${local.name}/anthropic-api-key" }
resource "aws_secretsmanager_secret" "github"    { name = "${local.name}/github-token" }
resource "aws_secretsmanager_secret" "slack"     { name = "${local.name}/slack-webhook-url" }

###############################################################################
# IAM
###############################################################################
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Least privilege: this role can read these three secrets and nothing else.
resource "aws_iam_role_policy" "execution_secrets" {
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [
        aws_secretsmanager_secret.anthropic.arn,
        aws_secretsmanager_secret.github.arn,
        aws_secretsmanager_secret.slack.arn,
      ]
    }]
  })
}

# The task role is empty on purpose: the app calls no AWS APIs itself.
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

###############################################################################
# Networking
###############################################################################
resource "aws_security_group" "alb" {
  name   = "${local.name}-alb"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name   = "${local.name}-svc"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]  # only the ALB may reach tasks
  }
  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    cidr_blocks = ["0.0.0.0/0"]  # outbound to the LLM, GitHub, Slack
  }
}

resource "aws_lb" "this" {
  name               = local.name
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "this" {
  name        = local.name
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/healthz"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 15
    timeout             = 5
  }

  # Feedback runs can take tens of seconds; let in-flight work finish on deploy.
  deregistration_delay = 60
}

###############################################################################
# Logs and compute
###############################################################################
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = 30
}

resource "aws_ecs_cluster" "this" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = local.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "api"
    image     = local.image
    essential = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    environment = [
      { name = "ENVIRONMENT", value = var.environment },
      { name = "LOG_FORMAT", value = "json" },
      { name = "LOG_LEVEL", value = "INFO" },
      { name = "GITHUB_REPO", value = "acme/product" },
    ]

    # Secrets arrive as env vars at container start, resolved from Secrets
    # Manager by the execution role. They are never in the task definition.
    secrets = [
      { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic.arn },
      { name = "GITHUB_TOKEN",      valueFrom = aws_secretsmanager_secret.github.arn },
      { name = "SLACK_WEBHOOK_URL", valueFrom = aws_secretsmanager_secret.slack.arn },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "api"
      }
    }

    healthCheck = {
      command  = ["CMD-SHELL", "python -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3)\""]
      interval = 30
      timeout  = 5
      retries  = 3
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = local.name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = false   # egress via NAT; tasks are not internet-reachable
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.this.arn
    container_name   = "api"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true    # a failing deploy rolls itself back
  }

  lifecycle {
    # CI pushes new task definitions; Terraform owns the shape, not the version.
    ignore_changes = [task_definition]
  }
}

###############################################################################
# Alarms -- the metrics in observability.py are only useful if something watches
# them. CloudWatch covers infrastructure; app metrics go to AMP/Grafana.
###############################################################################
resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${local.name}-unhealthy-hosts"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  dimensions = {
    TargetGroup  = aws_lb_target_group.this.arn_suffix
    LoadBalancer = aws_lb.this.arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "server_errors" {
  alarm_name          = "${local.name}-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  dimensions = { LoadBalancer = aws_lb.this.arn_suffix }
}

output "alb_dns_name" { value = aws_lb.this.dns_name }
output "ecr_repository_url" { value = aws_ecr_repository.app.repository_url }
