import lit.formats

config.name = "Silly-GetTests"
config.test_format = lit.formats.ShTest(execute_external=True)

# Recognize .silly files as tests
config.suffixes = ['.silly']

# Use the generated site config
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, "..", "..", "..", "build", "tests", "lit", "get")

lit_config.load_config(
    config,
    os.path.join(config.test_source_root, "..", "..", "..", "build", "lit.site.cfg.py")
)
