#!/bin/sh
# Vendors a build of the chat UI into internal/webui.
#
#   scripts/sync-webui.sh <tarball path or https URL> <sha256> <version>
#
# The tarball holds index.html, chat.js and chat.css at its root. The script
# replaces internal/webui/dist with the tarball's contents and records the
# version and checksum in internal/webui/VERSION and SHA256. It changes
# nothing unless the tarball matches the checksum, has the three files, and
# keeps every member inside its root.
#
# WEBUI_DIR overrides the target directory (default: internal/webui next to
# this script's directory).
set -eu

if [ $# -ne 3 ]; then
	echo "usage: $0 <tarball path or https URL> <sha256> <version>" >&2
	exit 2
fi
source=$1
sha256=$2
version=$3

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
target=${WEBUI_DIR:-$root/internal/webui}

fail() {
	echo "sync-webui: $*" >&2
	exit 1
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

case $source in
https://*)
	tarball=$work/download
	curl --fail --silent --show-error --location --proto '=https' --output "$tarball" "$source" ||
		fail "could not download $source"
	;;
*://*)
	fail "refusing to download $source: only https URLs are accepted"
	;;
*)
	tarball=$source
	[ -f "$tarball" ] || fail "no such tarball: $tarball"
	;;
esac

actual=$(shasum -a 256 "$tarball" | cut -d ' ' -f 1)
[ "$actual" = "$sha256" ] || fail "checksum mismatch: expected $sha256, got $actual"

# Check the member names before unpacking anything, rather than trusting tar
# to refuse them.
tar -tf "$tarball" >"$work/names"
if grep -Eq '^/|(^|/)\.\.(/|$)' "$work/names"; then
	fail "the tarball has paths outside its root"
fi
# A link could lead outside the root under an innocent name.
if tar -tvf "$tarball" | grep -qv '^[-d]'; then
	fail "the tarball may hold only regular files and directories"
fi

mkdir "$work/dist"
tar -xf "$tarball" -C "$work/dist"
for required in index.html chat.js chat.css; do
	[ -f "$work/dist/$required" ] || fail "the tarball has no $required at its root"
done

mkdir -p "$target"
rm -rf "$target/dist"
mv "$work/dist" "$target/dist"
printf '%s\n' "$version" >"$target/VERSION"
printf '%s\n' "$sha256" >"$target/SHA256"
echo "sync-webui: vendored version $version into $target"
