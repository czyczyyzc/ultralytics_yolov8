# CM5 vendor 5.10.198 experiment environment

## Verified deployment (2026-09-21)

This is a board-specific deployment, not a portable kernel installer.
The target is the second RK3588S OPi CM5 with 32 GB eMMC:

- SSH: `ssh orangepi@192.168.144.50` (existing account/password unchanged).
- Mac USB Ethernet: `192.168.144.49/24`; do not assign the board's `.50` to the Mac.
- Running kernel: `5.10.198`, vendor build `#1150`.
- Userspace: existing Orange Pi Ubuntu 22.04, not the RAM-only diagnostic image.
- Root: `/dev/mmcblk0p1`, ext4, mounted read/write.
- Root UUID: `25606694-45b8-4ae4-abd1-ee4d5bed51fe`.
- eMMC CID: `4501004456343033320175a9ff04bc00`.
- Default boot was installed, followed by a successful reboot and SSH verification.
- `/proc/cmdline` contains `camera_vendor_persistent=1`.

Both the trial boot and persistent boot completed a 600-frame camera capture
with exit status 0 and reported 120.00 FPS. This is raw capture only, without
image conversion, resizing, inference, tracking, or display. It is not a long
thermal stability test.

Camera node: `/dev/video11`, `rkcif-mipi-lvds2`, 1920 x 1080, `BA81`,
2048-byte row stride, 2211840-byte buffer. The kernel identifies the sensor
driver as `sc233hgs 3-0030`. The raw format code alone does not establish whether
the physical sensor is monochrome or color.

```sh
timeout 20 v4l2-ctl -d /dev/video11 --stream-mmap=8 \
  --stream-count=600 --stream-to=/dev/null
```

## Files on the board

Staging and logs:
`/home/orangepi/camera_boot_stage/vendor_ubuntu_20260921/`

- `init`, `install.sh`, `trial.cmd`, `permanent.cmd`: deployment scripts.
- `trial-camera-600.log`, `trial-system.log`, `trial-dmesg.log`: trial evidence.
- `persistent-camera-600.log`, `persistent-system.log`, `persistent-dmesg.log`: post-reboot evidence.

Boot assets:

- `/boot/rk3588-camera.Image`: vendor kernel.
- `/boot/dtb/rockchip/viz-mr063M-camera.dtb`: matching camera DTB.
- `/boot/camera-vendor-ubuntu/initramfs.cpio.gz`: bridge to existing Ubuntu root.
- `/boot/camera-vendor-ubuntu/original/`: original boot.scr, boot.cmd and orangepiEnv.txt.
- `/boot/boot.scr`: persistent vendor boot script.

The original 6.1 kernel and DTB remain installed. Boot-script backup is not a
full-disk backup. The initramfs and staging rootfs contain host SSH keys and
password hashes: do not publish them or place them in a shared delivery bundle.
The normal Ubuntu SSH authentication configuration was not changed. A local
Mac public key was installed only in the temporary rescue rootfs.

## Return to the original kernel

Run from the working Ubuntu system on this board:

```sh
stage=/home/orangepi/camera_boot_stage/vendor_ubuntu_20260921
sudo bash "$stage/install.sh" restore "$stage"
sudo reboot
```

This selects the retained `6.1.43-rockchip-rk3588` boot configuration, where the
camera was not available. It is not executed automatically after experiments.
If SSH is unavailable, recovery needs a serial/bootloader console or another
way to mount this eMMC and restore the saved boot files.

## Safety and limitations

- `trial` restores the original boot block before entering the vendor kernel.
  Its raw block address is guarded against this exact board and file layout.
  Do not reuse it on another board or after file extents change.
- `persist` has no automatic rollback after a kernel hang. Its fallback only
  handles boot asset loading failure or a returned boot command.
- `rebuild` updates an existing prepared initramfs while original boot is selected;
  it does not select a kernel. Its checks deliberately refuse a changed layout.
- The initial trial falsely rejected `/sbin/init`, an absolute symlink. The
  corrected check runs inside the Ubuntu root using `chroot`; the board test passed.
- `/lib/modules/5.10.198` is absent. Built-in camera/network/NPU drivers load,
  but this is not a complete vendor module installation.
- `run-rpc_pipefs.mount` fails because `rpc_pipefs` is unavailable. NFS/RPC usage
  has not been validated.
- `dnsmasq` fails because port 53 is already occupied by `systemd-resolved`.
  These services were not disabled or reconfigured to hide the failure.
- The NPU driver reports `0.9.8`; RKNN model compatibility and inference FPS
  were not tested in this task. Driver initialization includes resource and
  power-model warnings, so successful enumeration is not an inference test.
- The board clock is stale. Log dates may show September 7; the host verification
  date is September 21. Do not infer experiment order solely from board timestamps.
- Avoid kernel/boot package upgrades during experiments: they may rewrite the
  selected boot files. No packages have been held automatically.

Local evidence copies are in `deliverables/cm5_vendor_ubuntu_20260921/`.
