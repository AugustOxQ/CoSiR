"""
Test to reproduce the cache reference bug
Created: 2025-01-13
"""

import torch
from collections import OrderedDict


class SimpleCacheTest:
    """Simplified cache to demonstrate the bug"""
    def __init__(self):
        self.cache = OrderedDict()

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        # Bug: This removes the old value first
        if key in self.cache:
            old = self.cache.pop(key)
            print(f"  Removed old value: {id(old)}")

        self.cache[key] = value
        print(f"  Put new value: {id(value)}")


def test_reference_bug():
    print("=" * 80)
    print("Testing Cache Reference Bug")
    print("=" * 80)

    cache = SimpleCacheTest()

    # Initial put
    print("\n1. Initial put")
    data = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    cache.put("key1", data)
    print(f"   Data object ID: {id(data)}")

    # Get, modify, and put back (BUGGY PATTERN)
    print("\n2. Get, modify, and put back - BUGGY")
    retrieved = cache.get("key1")
    print(f"   Retrieved object ID: {id(retrieved)}")
    print(f"   Same object? {id(retrieved) == id(data)}")

    # Modify it
    print("\n3. Modifying the retrieved dict")
    retrieved["a"] = torch.tensor([10.0, 20.0])
    print(f"   Modified 'a' to: {retrieved['a']}")

    # Put it back - THIS IS THE BUG!
    print("\n4. Putting modified dict back (same reference)")
    cache.put("key1", retrieved)

    # Try to get it again
    print("\n5. Getting it back after put")
    final = cache.get("key1")
    print(f"   Final object ID: {id(final)}")
    print(f"   Value of 'a': {final['a']}")

    print("\n" + "=" * 80)
    print("CORRECT PATTERN: Clone before modifying")
    print("=" * 80)

    # Correct pattern
    print("\n6. Get and clone")
    retrieved2 = cache.get("key1")
    # Clone the dict before modifying
    cloned = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in retrieved2.items()}
    print(f"   Retrieved ID: {id(retrieved2)}, Cloned ID: {id(cloned)}")

    print("\n7. Modify the cloned version")
    cloned["a"] = torch.tensor([100.0, 200.0])

    print("\n8. Put the cloned version back")
    cache.put("key1", cloned)

    print("\n9. Verify")
    final2 = cache.get("key1")
    print(f"   Final value of 'a': {final2['a']}")


if __name__ == "__main__":
    test_reference_bug()
