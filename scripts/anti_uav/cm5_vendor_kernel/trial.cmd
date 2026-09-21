# This LBA is valid only after install.sh checks the original file's extent.
# Restore Ubuntu's boot script BEFORE attempting the vendor kernel.
setenv load_addr "0x9000000"
setenv restore_ok "0"
if load ${devtype} ${devnum} ${load_addr} ${prefix}camera-vendor-ubuntu/restore-block.bin; then
 if mmc dev ${devnum}; then
  if mmc write ${load_addr} 0x283030 0x8; then
   setenv restore_ok "1"
  fi
 fi
fi
if test "${restore_ok}" = "1"; then
 setenv bootargs "rdinit=/init console=ttyFIQ0,1500000 console=tty1 loglevel=5 panic=10 cma=128M net.ifnames=0 camera_vendor_trial=1"
 if load ${devtype} ${devnum} ${ramdisk_addr_r} ${prefix}camera-vendor-ubuntu/initramfs.cpio.gz; then
  setenv initrd_size ${filesize}
  if load ${devtype} ${devnum} ${kernel_addr_r} ${prefix}rk3588-camera.Image; then
   if load ${devtype} ${devnum} ${fdt_addr_r} ${prefix}dtb/rockchip/viz-mr063M-camera.dtb; then
    booti ${kernel_addr_r} ${ramdisk_addr_r}:${initrd_size} ${fdt_addr_r}
   fi
  fi
 fi
fi
echo "Vendor Ubuntu trial failed; power cycle returns to original Ubuntu after successful restoration."
