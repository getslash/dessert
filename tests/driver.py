import sys

if __name__ == "__main__":
    [_, example_name] = sys.argv # pylint: disable=unbalanced-tuple-unpacking
    import dessert

    with dessert.rewrite_assertions_context():
        import examples  # pylint: disable=import-error

    func = getattr(examples, example_name)
    try:
        func()
    except AssertionError as e:
        print(e)
    else:
        sys.exit('Assertion not matched')
