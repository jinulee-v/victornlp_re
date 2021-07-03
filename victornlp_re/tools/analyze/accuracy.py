"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

from . import register_analysis_fn

@register_analysis_fn('accuracy')
def analyze_accuracy(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py'' for more details.
  
  @return Dictionary with accuracy(total) and per-label in percentage(rounded for 4 digits).
  """
