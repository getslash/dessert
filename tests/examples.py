def no_message_simple_equality():
    i = 1
    j = 2

    assert i == j

def with_message_simple_equality():
    i = 1
    j = 2

    assert i == j, 'Here is an assertion message'
