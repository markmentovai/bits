#!/bin/bash

if [[ ${#} -ne 2 ]]; then
  echo "usage: ${0} '/path/to/Install macOS.app' '/path/to/installer.vmdk'" >& 2
  exit 1
fi

set -ex -o pipefail

# createinstallmedia doesn’t clean up after itself. It leaves a device node
# hanging around, associated with the SharedSupport volume that it copied things
# out of. It does this regardless of success or failure. Bad tool!
# save_old_devices and detach_new_device are intended to clean up after it.

_devices() {
  diskutil list | grep ^/ | cut -d ' ' -f 1 | sort
}

save_old_devices() {
  _devices > "${WORK_DIR}/devices.0"
}

_new_device() {
  set +e
  _devices > "${WORK_DIR}/devices.1"
  diff "${WORK_DIR}/devices.0" "${WORK_DIR}/devices.1" |
      sed -En -e 's/^> (.*)/\1/p' > "${WORK_DIR}/devices.diff"
  COUNT=$(wc -l < "${WORK_DIR}/devices.diff")
  if [[ ${COUNT} -ne 1 ]]; then
    echo "${0}: warning: incorrect number of new devices" >& 2
    set -e
    exit 0
  fi
  set -e
  cat "${WORK_DIR}/devices.diff"
}

detach_new_device() {
  device=$(_new_device)
  if [[ -n "${device}" ]]; then
    sudo hdiutil detach "${device}"
  fi
}

SOURCE=${1}
DESTINATION=${2}

WORK_DIR=$(mktemp -d -t make_installer_vmdk)
trap 'set +e; rm -rf "${WORK_DIR}"' EXIT

hdiutil create -type SPARSE -size 16GB -fs 'Journaled HFS+' \
    -volname 'Install macOS' -o "${WORK_DIR}/installer.sparseimage"

mkdir "${WORK_DIR}/installer"
DEVICE=$(hdiutil attach -mountpoint "${WORK_DIR}/installer" -nobrowse \
    -noautofsck -noautoopen -noverify "${WORK_DIR}/installer.sparseimage" |
    head -1 | cut -d ' ' -f 1)
trap 'set +e; hdiutil detach "${DEVICE}"; rm -rf "${WORK_DIR}"' EXIT

save_old_devices
trap 'set +e;
      detach_new_device;
      hdiutil detach "${DEVICE}";
      rm -rf "${WORK_DIR}"' EXIT

# Note: --downloadassets could be used here, but it probably doesn’t offer
# anything that would be useful to a virtual machine.
sudo "${SOURCE}/Contents/Resources/createinstallmedia" \
    --volume "${WORK_DIR}/installer" --nointeraction

# 20211109: when working with 11.6.1 installer under 12.0.1, detach_new_device
# fails because it detects 2 new devices instead of 1, and hdiutil detach
# "${DEVICE}" fails with “Resource busy”. Probably the “busy resource” problem
# is also the cause of detach_new_device seeing too many devices if
# createinstallmedia began an unmount that didn’t complete as quickly as it used
# to because of the “busy” problem. Adding this “sleep” seems to let things
# settle down enough that the hdiutil detach operations succeed.
sleep 1
detach_new_device
trap 'set +e; hdiutil detach "${DEVICE}"; rm -rf "${WORK_DIR}"' EXIT

# 20211109: when working with 11.6.1 installer under 12.0.1, hdiutil detach
# "${DEVICE}" fails with "Resource busy". This "busy resource" problem may also
# be why too many new devices appeared above in detach_new_device, if something
# that had previously been unmounted now isn't able to before detach_new_device
# runs because it's "busy". Temporarily bypass this detach, and make it the
# user’s problem to clean up.
hdiutil detach "${DEVICE}"
trap 'set +e; rm -rf "${WORK_DIR}"' EXIT

hdiutil convert -format UDTO "${WORK_DIR}/installer.sparseimage" \
    -o "${WORK_DIR}/installer.cdr"

rm "${WORK_DIR}/installer.sparseimage"

mv "${WORK_DIR}/installer.cdr" "${WORK_DIR}/installer.iso"

qemu-img convert -f raw -O vmdk "${WORK_DIR}/installer.iso" "${DESTINATION}"

rm "${WORK_DIR}/installer.iso"
