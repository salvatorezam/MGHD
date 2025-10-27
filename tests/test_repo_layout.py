def test_imports_smoke():
    import importlib
    assert importlib.import_module("mghd.core")
    assert importlib.import_module("mghd.codes.registry")
    assert importlib.import_module("mghd.samplers")
    assert importlib.import_module("mghd.decoders")
    assert importlib.import_module("mghd.cli")


def test_sampler_registry_defaults():
    from mghd.samplers import get_sampler
    from mghd.samplers.cudaq_sampler import CudaQSampler
    cudaq = get_sampler("cudaq")
    assert isinstance(cudaq, CudaQSampler)
