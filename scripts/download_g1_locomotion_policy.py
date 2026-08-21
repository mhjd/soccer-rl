import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "g1_locomotion" / "policy.onnx"
UPSTREAM_COMMIT = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
POLICY_URL = (
    "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/"
    f"{UPSTREAM_COMMIT}/deploy/robots/g1_29dof/config/policy/velocity/"
    "v0/exported/policy.onnx"
)
EXPECTED_SHA256 = (
    "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the pinned official G1 locomotion policy.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()

    if output.exists() and not args.force:
        if sha256(output) == EXPECTED_SHA256:
            print(f"Policy already verified: {output}")
            return
        raise RuntimeError(
            f"Existing policy has an unexpected checksum: {output}. "
            "Use --force to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix="g1-policy-",
        suffix=".onnx",
        dir=output.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        with urlopen(POLICY_URL) as response, temporary_path.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)

        actual_sha256 = sha256(temporary_path)
        if actual_sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded policy checksum mismatch: "
                f"expected {EXPECTED_SHA256}, got {actual_sha256}"
            )
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Downloaded and verified policy: {output}")


if __name__ == "__main__":
    main()
