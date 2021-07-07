"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

from . import register_analysis_fn

@register_analysis_fn('mtre')
def analyze_mtre_basic(inputs):
  """
  Calculates UAS for DP; recovery for entity type labeling, and Accuracy for relation labeling.
  
  @param inputs List of dictionaries. Refer to 'dataset.py'' for more details.
  
  @return Dictionary with accuracy(total) and per-label in percentage(rounded for 4 digits).
  """
  dp_correct = 0
  dp_total = 0
  
  etl_correct = 0
  etl_total = 0
  
  re_correct = 0
  re_total = 0

  for input in inputs:
    # UAS
    assert 'dependency' in input and 'dependency_predict' in input
    assert len(input['dependency']) == len(input['dependency_predict'])
    for arc_golden, arc_predict in zip(input['dependency'], input['dependency_predict']):
      assert arc_golden['dep'] == arc_predict['dep']
      dp_total += 1
      if arc_golden['head'] == arc_predict['head']:
        dp_correct += 1
    
    # Entity Type labeling
    for entity in input['named_entity']:
      assert 'label' in entity and 'label_predict' in entity
      etl_total += len(entity['label'])
      for predict in entity['label_predict']:
        if predict in entity['label']:
          etl_correct += 1

  dp_uas = dp_correct / dp_total * 100
  etl_recovery = etl_correct / etl_total * 100

  return {
    'UAS': round(dp_uas, 2),
    'entity_type_label': {
      'recovery': round(etl_recovery, 2)
    }
  }


