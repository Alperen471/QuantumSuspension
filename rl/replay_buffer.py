"""
rl/replay_buffer.py — QFM pre-encode desteği ile replay buffer.

QFM modu: encoder verilirse state'ler add() sırasında encode edilir.
Training'de sıfır quantum circuit call → baseline kadar hızlı.
"""
import numpy as np
import torch
from typing import NamedTuple


class Batch(NamedTuple):
    states     : torch.Tensor
    actions    : torch.Tensor
    rewards    : torch.Tensor
    next_states: torch.Tensor
    dones      : torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim=5, action_dim=1,
                 capacity=100_000, device="cuda", encoder=None):
        self.capacity  = capacity
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder   = encoder
        self._ptr      = 0
        self._size     = 0

        if encoder is not None and hasattr(encoder, "output_dim"):
            store_dim      = encoder.output_dim
            self._qfm_mode = True
        else:
            store_dim      = state_dim
            self._qfm_mode = False

        self._store_dim = store_dim
        self._state_dim = state_dim

        self.states      = np.zeros((capacity, store_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, store_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

        if self._qfm_mode:
            pass  # Pre-encode modu devre dışı

    def _enc(self, state: np.ndarray) -> np.ndarray:
        if not self._qfm_mode:
            return state
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0)
            return self.encoder.encode_torch(t).squeeze(0).cpu().numpy().astype(np.float32)

    def add(self, state, action, reward, next_state, done):
        state      = np.asarray(state,      dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        # Normal QFM buffer modu
        s_enc  = self._enc(state)
        ns_enc = self._enc(next_state)

        # Pre-encoded buffer: yeni env state'leri ham (5D) gelir
        # _live_encoder ile encode et
        if s_enc.shape[0] != self._store_dim:
            live_enc = getattr(self, "_live_encoder", None)
            if live_enc is not None:
                import torch
                with torch.no_grad():
                    s_enc  = live_enc.encode_torch(
                        torch.FloatTensor(state).unsqueeze(0)
                    ).squeeze(0).cpu().numpy().astype(np.float32)
                    ns_enc = live_enc.encode_torch(
                        torch.FloatTensor(next_state).unsqueeze(0)
                    ).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                # Fallback: sıfır pad
                s_enc  = np.zeros(self._store_dim, dtype=np.float32)
                ns_enc = np.zeros(self._store_dim, dtype=np.float32)

        self.states[self._ptr]      = s_enc
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = ns_enc
        self.dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size=256) -> Batch:
        idx = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            states      = torch.FloatTensor(self.states[idx]).to(self.device),
            actions     = torch.FloatTensor(self.actions[idx]).to(self.device),
            rewards     = torch.FloatTensor(self.rewards[idx]).to(self.device),
            next_states = torch.FloatTensor(self.next_states[idx]).to(self.device),
            dones       = torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def add_batch(self, npz_data: dict) -> None:
        """
        Toplu yükleme.
        QFM modunda encoder.encode_torch(chunk) çağrısı yapılır —
        bu çağrı içinde ThreadPoolExecutor paralel encode çalışır.
        """
        states      = npz_data["states"]
        actions     = npz_data["actions"]
        rewards     = npz_data["rewards"]
        next_states = npz_data["next_states"]
        dones       = npz_data["dones"]
        n = min(len(states), self.capacity)

        if False:  # QFM pre-encode devre dışı — live encode kullanılıyor
            import torch, os
            chunk = 256
            total_chunks = (n + chunk - 1) // chunk
            print(f"  [QFM Buffer] {n:,} state paralel encode ediliyor "
                  f"(chunk={chunk}, threads={min(chunk, os.cpu_count() or 4, 8)})...")

            for ci, start in enumerate(range(0, n, chunk)):
                end = min(start + chunk, n)
                with torch.no_grad():
                    sf  = self.encoder.encode_torch(
                        torch.FloatTensor(states[start:end].astype(np.float32))
                    ).cpu().numpy()
                    nsf = self.encoder.encode_torch(
                        torch.FloatTensor(next_states[start:end].astype(np.float32))
                    ).cpu().numpy()

                for i in range(end - start):
                    p = self._ptr % self.capacity
                    self.states[p]      = sf[i]
                    self.actions[p]     = actions[start + i]
                    self.rewards[p]     = rewards[start + i]
                    self.next_states[p] = nsf[i]
                    self.dones[p]       = float(dones[start + i])
                    self._ptr  = (self._ptr + 1) % self.capacity
                    self._size = min(self._size + 1, self.capacity)

                if ci % 20 == 0 or ci == total_chunks - 1:
                    pct = min(end, n) / n * 100
                    print(f"    {pct:5.1f}%  ({min(end,n):,}/{n:,})", end="\r")

            print(f"\n  [QFM Buffer] Tamamlandı: {self._size:,} encoded state hazır")
        else:
            for i in range(n):
                self.add(states[i], actions[i], rewards[i],
                         next_states[i], float(dones[i]))

    @property
    def size(self): return self._size
    def is_ready(self, bs=256): return self._size >= bs
    @property
    def is_qfm(self): return self._qfm_mode
    def __len__(self): return self._size
    def __repr__(self):
        return (f"ReplayBuffer(size={self._size}/{self.capacity}, "
                f"mode={'QFM' if self._qfm_mode else 'std'}, "
                f"device={self.device})")
