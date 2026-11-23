variable "region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "sim-a"
}

variable "subnet_id" {
  description = "Single subnet in the AZ that holds the capacity reservation. EFA traffic does not cross AZs."
  type        = string
}

variable "security_group_id" {
  description = "Must allow all traffic to and from itself; EFA's SRD needs it."
  type        = string
}

variable "ami_id" {
  description = "Deep Learning Base GPU AMI (Ubuntu 22.04) pinned by ID, so driver/CUDA match the golden manifest."
  type        = string
}

variable "capacity_reservation_id" {
  description = "ODCR or Capacity Block for ML reservation id for the p5 nodes."
  type        = string
}

variable "node_count" {
  type    = number
  default = 4
}

variable "bringup_manifest_s3" {
  description = "s3:// URL of the golden manifest the node hook gates against."
  type        = string
}

variable "bringup_results_s3" {
  description = "s3:// prefix where each node uploads its captures and verdict."
  type        = string
}
