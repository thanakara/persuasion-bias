import readline  # noqa: F401

from persuasion_bias.utils import Turn


def chat(agent):
    while True:
        try:
            input_ = input(f"{Turn.USER}: ")
            response = agent(input_)
            print(f"{Turn.ASSISTANT}: {response}")
        except KeyboardInterrupt as e:
            print(f"\nExiting from: {e.__class__.__name__}")
            break
