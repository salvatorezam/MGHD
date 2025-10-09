def test_imports_smoke():
    import importlib
    assert importlib.import_module("core")
    assert importlib.import_module("codes_registry")
    assert importlib.import_module("samplers")
    assert importlib.import_module("teachers")
    assert importlib.import_module("tools")


def test_sampler_registry_defaults():
    from samplers import get_sampler, CudaQSampler
    cudaq = get_sampler("cudaq")
    assert isinstance(cudaq, CudaQSampler)
