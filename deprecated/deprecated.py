def prepare_data(data, pipeline=('load_tiff', 'crop_tissue')):
    for func in pipeline:
        if isinstance(func, str):
            func = getattr(sys.modules[__name__], func)
        if not isinstance(data, tuple):
            data = (data,)
        print(func)
        print(data)
        data = func(*data)
    return data