output "asg_name" {
  value = aws_autoscaling_group.p5.name
}

output "placement_group" {
  value = aws_placement_group.gpu.name
}
