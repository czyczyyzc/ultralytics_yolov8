#!/usr/bin/env bash
# Deliberately restricted to the previously audited 32GB eMMC CM5.
set -euo pipefail
mode=${1:?prepare, trial, persist, or restore required}
stage=$(realpath "${2:?staging directory required}")
boot=/boot/camera-vendor-ubuntu
original_sha=7632c8e9f07eebc2dbb477b360ca4993f6fe3ca84e7a2140dbf8ec27ed618192
test "$EUID" -eq 0
test "$(findmnt -n -o SOURCE /)" = /dev/mmcblk0p1
test "$(findmnt -n -o FSTYPE /)" = ext4
test "$(cat /sys/class/block/mmcblk0/device/cid)" = 4501004456343033320175a9ff04bc00
test "$(blkid -s UUID -o value /dev/mmcblk0p1)" = 25606694-45b8-4ae4-abd1-ee4d5bed51fe
sha() { sha256sum "$1" | awk '{print $1}'; }
original() { test "$(sha /boot/boot.scr)" = "$original_sha"; }
extents() {
    test "$(cat /sys/class/block/mmcblk0p1/start)" = 61440
    test "$(stat -c %s /boot/boot.scr)" = 3413
    test "$(filefrag -v /boot/boot.scr | awk '$1 == "0:" {printf "%d %d", $4, $6}')" = '321542 1'
}
assets() {
    test "$(sha /boot/rk3588-camera.Image)" = f69a10e38ef0ae86b916edeebf5ed354697371f8af675dc5a1df130ae1e4f6f4
    test "$(sha /boot/dtb/rockchip/viz-mr063M-camera.dtb)" = fad0214d337e12bcaf4729dbb1849abed8384ee83b3aa0ea6d30c105efc0d865
}
case "$mode" in
prepare)
    original; extents; assets
    test ! -e "$boot"
    test ! -e "$stage/rootfs"
    mkdir -m 700 "$boot"
    mkdir -m 700 "$boot/original"
    cp -p /boot/boot.cmd /boot/boot.scr /boot/orangepiEnv.txt "$boot/original/"
    dd if=/dev/mmcblk0 of="$boot/restore-block.bin" bs=512 skip=$((0x283030)) count=8 status=none
    head -c 3413 "$boot/restore-block.bin" | cmp - /boot/boot.scr
    cp -a /home/orangepi/camera_boot_stage/live_diag_20260907/rootfs "$stage/rootfs"
    root="$stage/rootfs"
    for name in switch_root blkid; do
        tool=$(command -v "$name")
        install -D "$tool" "$root$tool"
        while read -r lib; do
            install -D "$lib" "$root$lib"
        done < <(ldd "$tool" | awk '$2 == "=>" && $3 ~ /^\// {print $3} $1 ~ /^\// {print $1}')
    done
    install -m 600 /etc/ssh/ssh_host_ed25519_key "$root/etc/ssh/ssh_host_ed25519_key"
    install -m 755 "$stage/init" "$root/init"
    chroot "$root" /bin/sh -n /init
    chroot "$root" /usr/sbin/switch_root --help >/dev/null
    (cd "$root" && find . -print0 | cpio --null -o --format=newc | gzip -1) > "$boot/initramfs.cpio.gz"
    chmod 600 "$boot/initramfs.cpio.gz" "$boot/restore-block.bin"
    gzip -t "$boot/initramfs.cpio.gz"
    mkimage -C none -A arm -T script -d "$stage/trial.cmd" "$stage/trial.scr"
    mkimage -C none -A arm -T script -d "$stage/permanent.cmd" "$stage/permanent.scr"
    test "$(stat -c %s "$stage/trial.scr")" -le 3413
    cp -p "$stage/permanent.cmd" "$stage/permanent.scr" "$boot/"
    sha256sum "$boot/initramfs.cpio.gz" "$boot/original/boot.scr" > "$boot/SHA256SUMS"
    sync
    original; extents
    echo 'Prepared and backed up. Boot selection has not changed.'
    ;;
trial)
    original; extents; assets
    test "$(sha "$boot/original/boot.scr")" = "$original_sha"
    head -c 3413 "$boot/restore-block.bin" | cmp - /boot/boot.scr
    sha256sum -c "$boot/SHA256SUMS"
    test "$(stat -c %s "$stage/trial.scr")" -le 3413
    dd if="$stage/trial.scr" of=/boot/boot.scr conv=notrunc,fsync status=none
    head -c "$(stat -c %s "$stage/trial.scr")" /boot/boot.scr | cmp - "$stage/trial.scr"
    extents
    sync
    echo 'One-shot trial armed. Reboot explicitly; old boot script is restored before kernel entry.'
    ;;
persist)
    original; assets
    test "$(uname -r)" = 5.10.198
    test "$(cat /run/camera-vendor/root-verified)" = 5.10.198
    test -c /dev/video11
    test "$(sha "$boot/original/boot.scr")" = "$original_sha"
    sha256sum -c "$boot/SHA256SUMS"
    install -m 644 "$boot/permanent.cmd" /boot/boot.cmd.vendor-new
    install -m 644 "$boot/permanent.scr" /boot/boot.scr.vendor-new
    mv /boot/boot.cmd.vendor-new /boot/boot.cmd
    mv /boot/boot.scr.vendor-new /boot/boot.scr
    sync
    cmp /boot/boot.scr "$boot/permanent.scr"
    echo 'Persistent vendor kernel selected; old kernel, DTB and Ubuntu root retained.'
    ;;
restore)
    test "$(sha "$boot/original/boot.scr")" = "$original_sha"
    install -m 644 "$boot/original/boot.cmd" /boot/boot.cmd.restore-new
    install -m 644 "$boot/original/boot.scr" /boot/boot.scr.restore-new
    mv /boot/boot.cmd.restore-new /boot/boot.cmd
    mv /boot/boot.scr.restore-new /boot/boot.scr
    sync
    original
    echo 'Original Ubuntu 6.1 boot selected for the next reboot.'
    ;;
*) echo 'Unknown mode' >&2; exit 2 ;;
esac
