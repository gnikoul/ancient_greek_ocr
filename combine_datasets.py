from pathlib import Path

EPARCHOS = Path("datasets/eparchos/training_nfd/train.lst")
XF53 = Path("datasets/XF53/train.lst")

OUTPUT = Path("datasets/combined_train.lst")


def main():
    eparchos = EPARCHOS.read_text(encoding="utf-8").splitlines()
    xf53 = XF53.read_text(encoding="utf-8").splitlines()

    combined = eparchos + xf53

    OUTPUT.write_text("\n".join(combined) + "\n", encoding="utf-8")

    print(f"EPARCHOS samples: {len(eparchos)}")
    print(f"XF53 samples:     {len(xf53)}")
    print(f"Combined samples: {len(combined)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()