import json
import argparse
import sys

class MondoOntologyMatcher:
    def __init__(self, json_path):
        """
        MONDOのJSONファイルを読み込み、OMIM IDからMONDO IDへのマッピングと
        MONDO内の親子関係（階層構造）を構築します。
        """
        self.omim_to_mondo = {}  # OMIM:ID -> set(MONDO:ID)
        self.mondo_to_parents = {}  # MONDO:ID -> set(Parent MONDO:IDs)
        self._load_mondo_json(json_path)

    def _load_mondo_json(self, path):
        print(f"Loading {path}...", file=sys.stderr)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error: ファイルの読み込みに失敗しました: {e}", file=sys.stderr)
            sys.exit(1)

        # 標準的なObograph形式のJSONを想定
        graph = data.get('graphs', [{}])[0]

        # 1. ノードの走査（OMIMマッピングの構築）
        for node in graph.get('nodes', []):
            mondo_id = node.get('id')
            if not mondo_id:
                continue

            meta = node.get('meta', {})
            xrefs = meta.get('xrefs', [])
            for xref in xrefs:
                val = xref.get('val', '')
                if val.startswith('OMIM:'):
                    if val not in self.omim_to_mondo:
                        self.omim_to_mondo[val] = set()
                    self.omim_to_mondo[val].add(mondo_id)

        # 2. エッジの走査（親子関係の構築）
        for edge in graph.get('edges', []):
            sub = edge.get('sub')  # 子
            obj = edge.get('obj')  # 親
            pred = edge.get('pred', '')

            # is_a関係（subClassOf）を抽出
            if 'subClassOf' in pred or pred == 'is_a':
                if sub not in self.mondo_to_parents:
                    self.mondo_to_parents[sub] = set()
                self.mondo_to_parents[sub].add(obj)
        print("Done loading.", file=sys.stderr)

    def _format_id(self, omim_id):
        """入力されたIDを 'OMIM:数字' の形式に整えます。"""
        s = str(omim_id).strip()
        if not s.startswith('OMIM:'):
            return f"OMIM:{s}"
        return s

    def judge_relationship(self, omim_id1, omim_id2):
        """
        二つのOMIM IDの関係性を判定します。
        """
        id1 = self._format_id(omim_id1)
        id2 = self._format_id(omim_id2)

        mondo_ids1 = self.omim_to_mondo.get(id1, set())
        mondo_ids2 = self.omim_to_mondo.get(id2, set())

        if not mondo_ids1 or not mondo_ids2:
            missing = []
            if not mondo_ids1: missing.append(id1)
            if not mondo_ids2: missing.append(id2)
            return f"ERROR: MAPPING NOT FOUND for {', '.join(missing)}"

        # 1. MATCH: 同じMONDO IDを共有している場合
        if not mondo_ids1.isdisjoint(mondo_ids2):
            return "MATCH"

        # 2. CLOSE MATCH: 親子関係または兄弟関係の判定
        for m1 in mondo_ids1:
            for m2 in mondo_ids2:
                p1 = self.mondo_to_parents.get(m1, set())
                p2 = self.mondo_to_parents.get(m2, set())

                # 親子関係 (m1がm2の親、またはその逆)
                if m2 in p1 or m1 in p2:
                    return "CLOSE MATCH"

                # 兄弟関係 (共通の親を持っている)
                if not p1.isdisjoint(p2):
                    return "CLOSE MATCH"

        return "NO MATCH"

def main():
    parser = argparse.ArgumentParser(description='OMIM IDを用いてMONDO上での疾患の近さを判定します。')
    parser.add_argument('omim1', help='1つ目のOMIM ID (例: 122470)')
    parser.add_argument('omim2', help='2つ目のOMIM ID (例: 613659)')
    parser.add_argument('--json', default='mondo-base.json', help='MONDO JSONのパス')

    args = parser.parse_args()

    matcher = MondoOntologyMatcher(args.json)
    result = matcher.judge_relationship(args.omim1, args.omim2)
    
    print("-" * 30)
    print(f"OMIM 1: {args.omim1}")
    print(f"OMIM 2: {args.omim2}")
    print(f"RESULT: {result}")
    print("-" * 30)

if __name__ == "__main__":
    main()