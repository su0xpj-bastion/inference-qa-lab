# On-prem day-0: PXE / Base Command Manager notes

Ansible starts once a node has an OS and SSH. These are the steps before that, and the
evidence each one should leave for the acceptance pipeline.

| Step | How | Evidence / check |
|---|---|---|
| Rack & cable | Cable map per rack: rail *k* of every node to leaf *k* (rail-optimised). | `gpu_nic_affinity` and `fabric_ports` fail if a rail cable or NIC slot is swapped. |
| BMC reachable | DHCP reservation for each BMC MAC from the asset sheet; Redfish `GET /redfish/v1/Systems`. | Inventory serials match the asset sheet before power-on. |
| BIOS profile | Redfish `PATCH /Bios/Settings`: SR-IOV/ACS per vendor guide, PCIe gen set to Auto/Gen5 (never forced lower), performance power profile. | `pcie_link` catches a slot capped at gen4 via `max_host_link_gen`. |
| Firmware baseline | Vendor bundle (BMC, BIOS, GPU VBIOS / baseboard, CX-7 FW via `mlxfwmanager`) before OS install. | `software_versions` compares VBIOS and HCA firmware to the golden manifest. |
| PXE install | BCM: assign the node to the category for its node type (software image pinned to the golden kernel + driver). Without BCM: dnsmasq/iPXE + autoinstall (Ubuntu) / kickstart. | `uname-r.txt` vs golden `kernel`. |
| Fabric | UFM / OpenSM sees every port; partition keys applied. | `ibstat` Active/LinkUp at 400 Gb/s on every compute rail. |
| Hand-off | `ansible-playbook -i inventory.ini site.yml` then `acceptance.yml`. | `reports/onprem/bringup.{md,json}`. |

With BCM, `cmsh -c "device; list"` gives the node/category mapping that becomes the
`node_type` in `inventory.ini`; keep that the single source rather than editing both by hand.
