import json
import sys

class MondoOntologyMatcher:
    def __init__(self, json_path, child_threshold=50):
        self.omim_to_mondo = {}
        self.mondo_to_parents = {}
        self.parent_to_child_count = {}
        self.child_threshold = child_threshold
        self._load_mondo_json(json_path)

    def _load_mondo_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = data.get('graphs', [{}])[0]
        
        # 1. ノードの走査（正規化してマッピング）
            for node in graph.get('nodes', []):
                mondo_id = node.get('id')
                for xf in node.get('meta', {}).get('xrefs', []):
                    val = xf.get('val', '')
                    if 'OMIM:' in val.upper():
                    # "OMIM: 123456" や "omim:123" を "OMIM:123456" に統一
                        clean_val = 'OMIM:' + ''.join(filter(str.isdigit, val))
                        if mondo_id not in self.omim_to_mondo:
                            self.omim_to_mondo[mondo_id] = set()
                        self.omim_to_mondo[mondo_id].add(clean_val)

        # 2. エッジの走査（is_a を含む述語に対応）
            for edge in graph.get('edges', []):
                pred = edge.get('pred', '')
            # 'is_a' という文字列が含まれているかチェック（URL形式への対応）
                if 'is_a' in pred or pred.endswith('subClassOf'):
                    sub, obj = edge.get('sub'), edge.get('obj')
                    if sub and obj:
                        if sub not in self.mondo_to_parents:
                            self.mondo_to_parents[sub] = set()
                        self.mondo_to_parents[sub].add(obj)
                        self.parent_to_child_count[obj] = self.parent_to_child_count.get(obj, 0) + 1
        except Exception as e:
            print(f"Error loading Mondo JSON: {e}")

    def get_mondo_ids(self, omim_id):
        # 'OMIM:123' 形式で検索
        target = omim_id if omim_id.startswith('OMIM:') else f"OMIM:{omim_id}"
        return [m for m, omims in self.omim_to_mondo.items() if target in omims]

    def judge(self, omim1, omim2):
        if not omim1 or not omim2: return "NO MATCH"
        if str(omim1) == str(omim2): return "MATCH"

        m1_list = self.get_mondo_ids(omim1)
        m2_list = self.get_mondo_ids(omim2)

        for m1 in m1_list:
            for m2 in m2_list:
                if m1 == m2: return "MATCH"
                p1 = self.mondo_to_parents.get(m1, set())
                p2 = self.mondo_to_parents.get(m2, set())
                # 親子関係
                if m2 in p1 or m1 in p2: return "CLOSE MATCH"
                # 兄弟関係
                common = p1.intersection(p2)
                for cp in common:
                    if self.parent_to_child_count.get(cp, 0) <= self.child_threshold:
                        return "CLOSE MATCH"
        return "NO MATCH"