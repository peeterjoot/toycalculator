import lit.formats

config.name = "Silly-DriverTests"
config.test_format = lit.formats.ShTest(execute_external=True)

# Recognize .silly files as tests
config.suffixes = ['.silly', '.ll', '.mlir']

# Exclude support/module files, that are only built with other MAINs, so they aren't discovered as standalone tests
config.excludes = ['mod1.silly', 'mod2.silly', 'mod3.silly']

# Use the generated site config
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, "..", "..", "..", "build", "tests", "lit", "driver")

lit_config.load_config(
    config,
    os.path.join(config.test_source_root, "..", "..", "..", "build", "lit.site.cfg.py")
)
