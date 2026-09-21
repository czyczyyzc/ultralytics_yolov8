# Persistent camera kernel + the original Ubuntu root, not a RAM-only system.
setenv bootargs "rdinit=/init console=ttyFIQ0,1500000 console=tty1 loglevel=5 panic=10 cma=128M net.ifnames=0 camera_vendor_persistent=1"
if load ${devtype} ${devnum} ${ramdisk_addr_r} ${prefix}camera-vendor-ubuntu/initramfs.cpio.gz; then
 setenv initrd_size ${filesize}
 if load ${devtype} ${devnum} ${kernel_addr_r} ${prefix}rk3588-camera.Image; then
  if load ${devtype} ${devnum} ${fdt_addr_r} ${prefix}dtb/rockchip/viz-mr063M-camera.dtb; then
   booti ${kernel_addr_r} ${ramdisk_addr_r}:${initrd_size} ${fdt_addr_r}
  fi
 fi
fi
echo "Vendor files unavailable; trying the retained original Ubuntu boot script"
setenv load_addr "0x9000000"
if load ${devtype} ${devnum} ${load_addr} ${prefix}camera-vendor-ubuntu/original/boot.scr; then
 source ${load_addr}
fi
