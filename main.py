from interface import MainInterface


if __name__ == "__main__":
    # Launch Gradio interface
    print("\n" + "=" * 50)
    print("Launching Main Interface")
    print("=" * 50)
    print("\nThe interface will open in your default browser.")
    print("If it doesn't open automatically, click the URL shown below.")
    print("\nIn the interface, you can:")
    print("1. Choose role (non-expert or expert), enter questions, and task for the most optimized legal decision-making process")
    print("2. Review and modify generated arguments interactively")
    print("3. Get an accuracy answer and faithful explanations")
    print("\nPress Ctrl+C to stop the server.\n")

    interface = MainInterface()
    interface.launch(share=False)  # Set share=True to create a public link