# Speed Optimization Recommendations

## 🔥 Priority 1: Equity Calculator Caching (30-50% speedup)

### Problem
`get_8_features()` is called for EVERY observation (reset, step, hand_over)
- Monte Carlo simulation is computationally expensive
- Same hand situation can be calculated multiple times

### Solution Options

#### Option A: LRU Cache (Easy, 20-30% speedup)
```python
# poker_rl/utils/equity_calculator.py
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_8_features_cached(hole_tuple, board_tuple, street):
    """Cached version - convert to tuples for hashability"""
    hole_cards = [Card(suit, rank) for (suit, rank) in hole_tuple]
    board = [Card(suit, rank) for (suit, rank) in board_tuple]
    return get_8_features(hole_cards, board, street)

# In obs_builder.py
hole_tuple = tuple((c.suit, c.rank) for c in hole_cards_original)
board_tuple = tuple((c.suit, c.rank) for c in board_original)
features = get_8_features_cached(hole_tuple, board_tuple, game.street.value)
```

#### Option B: Reduce Monte Carlo Samples (Easy, 40-50% speedup)
```python
# poker_rl/utils/equity_calculator.py
# Change MC_SAMPLES from current value to lower
MC_SAMPLES = 100  # Instead of 1000 (if currently 1000)
```
**Trade-off**: Slightly less accurate equity estimates

#### Option C: Pre-compute Common Situations (Hard, Best accuracy)
- Pre-compute all preflop situations
- Pre-compute common flop/turn textures
- Only MC simulate rare situations

---

## ⚡ Priority 2: Increase Workers (Immediate, 3-4x speedup)

### Current
```python
.env_runners(num_env_runners=4, ...)
```

### Recommended (train.py)
```python
import multiprocessing
num_cpus = multiprocessing.cpu_count()
optimal_workers = max(1, int(num_cpus * 0.8) - 1)  # Leave 1 for learner

.env_runners(num_env_runners=optimal_workers, ...)  # ~16 on 20-core CPU
```

**Expected**: 4x throughput (4 → 16 workers)

---

## 📊 Priority 3: Increase Batch Size (10-20% speedup)

### Current
```python
train_batch_size=32768
```

### Recommended (if GPU available)
```python
train_batch_size=65536  # or 131072 if GPU memory allows
```

Test with increasing sizes until GPU memory is full.

---

## 🔇 Priority 4: Disable Logging (5% speedup)

### Current
```python
self.should_log_this_hand = np.random.random() < 0.0001
```

### Recommended for speed
```python
self.should_log_this_hand = False  # Completely disable during training
```

---

## 📈 Expected Total Speedup

| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Base | 1.0x | 1.0x |
| + Equity Cache | 1.4x | 1.4x |
| + More Workers (4→16) | 4.0x | 5.6x |
| + Larger Batch | 1.15x | 6.4x |
| + No Logging | 1.05x | **6.7x** |

**Total Expected: 6-7x faster training speed!**

---

## 🎯 Quick Wins (Implement First)

1. ✅ Increase workers to 16
2. ✅ Add LRU cache to equity calculator
3. ✅ Increase batch size to 65536
4. ✅ Disable logging

**Implementation time**: 30 minutes  
**Expected speedup**: 5-7x
