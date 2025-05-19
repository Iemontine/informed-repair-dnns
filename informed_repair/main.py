import models

def main():
    network = models.vit_b_32(pretrained=True, eval=True)
    print(network)


if __name__ == "__main__":
    main()
