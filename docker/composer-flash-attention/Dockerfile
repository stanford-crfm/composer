FROM mosaicml/composer:0.10.0

RUN pip install --no-cache-dir transformers "datasets<2" 'braceexpand' 'zstandard' 'fsspec' "pyyaml>=5.4.1,<6" \
     --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install --no-cache-dir triton==1.0.0 timm==0.5.4 torchdata "pycocotools>=2.0.4"

RUN git clone https://github.com/HazyResearch/flash-attention.git
RUN cd flash-attention && python setup.py install

ENV PATH $PATH:/root/.local/bin:.local/bin
