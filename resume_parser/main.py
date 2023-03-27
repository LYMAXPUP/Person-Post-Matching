#!/usr/bin/env python
# -*- coding:utf-8 -*-

import hanlp


from .convert import transform_pdf
from .split import create_pdf_segments
from .extract import main


def get_all_info(resume_path):
    model = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
    info = {}
    if resume_path.endswith('pdf'):
        text_list = transform_pdf(resume_path)
        segment_dict = create_pdf_segments(text_list)
        info = main(text_list, segment_dict, model)
        print("ok")
    else:
        print('No function to transform!')

    return info

