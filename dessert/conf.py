class _Config(object):
    _message_introspection = True

    def is_message_introspection_enabled(self):
        return _Config._message_introspection

    def enable_message_introspection(self):
        _Config._message_introspection = True

    def disable_message_introspection(self):
        _Config._message_introspection = False

conf = _Config()
