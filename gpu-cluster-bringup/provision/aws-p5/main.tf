# AWS p5.48xlarge node group (8x H100, 32 EFA network cards) with a
# post-provisioning acceptance hook. A node comes up tagged bringup=pending,
# runs the node + domain stages from cloud-init, and tags itself accepted or
# cordoned. Burn-in (hours) runs afterwards as a scheduled job, not at boot.

locals {
  efa_cards = 32 # p5.48xlarge: 32 network cards, one EFA interface each
}

resource "aws_placement_group" "gpu" {
  name     = "${var.cluster_name}-p5"
  strategy = "cluster"
}

resource "aws_launch_template" "p5" {
  name_prefix   = "${var.cluster_name}-p5-"
  image_id      = var.ami_id
  instance_type = "p5.48xlarge"

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 2
  }

  capacity_reservation_specification {
    capacity_reservation_target {
      capacity_reservation_id = var.capacity_reservation_id
    }
  }

  placement {
    group_name = aws_placement_group.gpu.name
  }

  # Card 0 carries the primary ENA+EFA interface; cards 1..31 are EFA on device_index 1.
  dynamic "network_interfaces" {
    for_each = range(local.efa_cards)
    content {
      network_card_index    = network_interfaces.value
      device_index          = network_interfaces.value == 0 ? 0 : 1
      interface_type        = "efa"
      subnet_id             = var.subnet_id
      security_groups       = [var.security_group_id]
      delete_on_termination = true
    }
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 500
      volume_type = "gp3"
    }
  }

  user_data = base64encode(templatefile("${path.module}/cloud-init.yaml.tftpl", {
    cluster_name = var.cluster_name
    manifest_s3  = var.bringup_manifest_s3
    results_s3   = var.bringup_results_s3
    node_type    = "aws-p5-efa"
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Cluster = var.cluster_name
      bringup = "pending"
    }
  }
}

resource "aws_autoscaling_group" "p5" {
  name                = "${var.cluster_name}-p5"
  min_size            = var.node_count
  max_size            = var.node_count
  desired_capacity    = var.node_count
  vpc_zone_identifier = [var.subnet_id]
  placement_group     = aws_placement_group.gpu.id

  # A cordoned node must stay up for diagnosis (and an RMA ticket), not be
  # silently replaced, so ASG health checks are EC2-only and replacement is manual.
  health_check_type         = "EC2"
  health_check_grace_period = 3600
  protect_from_scale_in     = true

  launch_template {
    id      = aws_launch_template.p5.id
    version = "$Latest"
  }

  tag {
    key                 = "Cluster"
    value               = var.cluster_name
    propagate_at_launch = true
  }
}
