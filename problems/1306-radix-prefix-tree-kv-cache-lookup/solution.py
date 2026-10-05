# We will implement a radix prefix-tree, out of token-ids.
class TrieNode:
    def __init__(self):
        self.children = {}  # Maps token_id -> TrieNode
        self.is_leaf = False  # Marks the end of a complete inserted sequence


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens: list[int]) -> None:
        """
        Inserts a sequence of token IDs into the prefix tree.
        Marks the terminal node with is_leaf = True.
        """
        current_node = self.root
        for token in tokens:
            if token not in current_node.children:
                current_node.children[token] = TrieNode()
            current_node = current_node.children[token]

        current_node.is_leaf = True

    def lookup(self, tokens: list[int]) -> int:
        """
        Walks the token sequence starting from the root and returns 
        the length of the longest matching prefix stored in the tree.
        """
        current_node = self.root
        matched_len = 0

        for token in tokens:
            if token in current_node.children:
                matched_len += 1
                current_node = current_node.children[token]
            else:
                break

        return matched_len

def radix_prefix_cache(ops):
    """
    Process insert/lookup ops on a token-id prefix tree.

    Returns a list of longest-prefix match lengths, one per lookup.
    """
    trie = Trie()
    results = [ ]
    for op, tokens in ops:
        if op == "insert":
            trie.insert(tokens)
        elif op == "lookup":
            matched_len = trie.lookup(tokens)
            results.append(matched_len)

    return results
