import numpy as np

class MmWaveMARLEnv:
    def __init__(self, num_ue_beams=16, num_gnb_beams=64, history_len=8):
        self.num_ue_beams = num_ue_beams
        self.num_gnb_beams = num_gnb_beams
        self.history_len = history_len
        self.reset()

    def reset(self):
        self.t = 0
        self.ue_position = np.array([0.0, 0.0])
        self.ue_velocity = np.array([15.0, 2.0])
        self.blocker_pos = np.array([20.0, 5.0])
        self.h_history = np.zeros((self.history_len, self.num_gnb_beams), dtype=np.complex64)
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.h_history.real.flatten(), self.h_history.imag.flatten()])

    def step(self, action_ue, action_gnb):
        self.t += 1
        self.ue_position += self.ue_velocity * 0.001
        
        dist_to_gnb = np.linalg.norm(self.ue_position - np.array([50.0, 50.0]))
        is_blocked = self._check_blockage()
        
        base_loss = 32.4 + 20 * np.log10(28.0e9) + 30 * np.log10(dist_to_gnb)
        if is_blocked:
            base_loss += 35.0
            
        spatial_misalignment = np.abs(action_gnb - self._get_optimal_beam_index())
        beam_gain = 20.0 * np.sinc(spatial_misalignment / 4.0)
        
        received_power = 23.0 - base_loss + beam_gain
        noise_floor = -174.0 + 10 * np.log10(100.0e6) + 7.0
        sinr = received_power - noise_floor
        sinr_linear = 10**(sinr / 10.0)
        
        throughput = np.log2(1.0 + sinr_linear)
        feedback_penalty = 0.5 if action_ue == 1 else 0.0
        beam_drop_penalty = 5.0 if sinr < -3.0 else 0.0
        
        reward = throughput - feedback_penalty - beam_drop_penalty
        self._evolve_channel(is_blocked)
        
        return self._get_state(), reward, (sinr < -5.0)

    def _get_optimal_beam_index(self):
        angle = np.arctan2(50.0 - self.ue_position[1], 50.0 - self.ue_position[0])
        return int(((angle + np.pi) / (2 * np.pi)) * self.num_gnb_beams) % self.num_gnb_beams

    def _check_blockage(self):
        return 15.0 < self.ue_position[0] < 25.0

    def _evolve_channel(self, is_blocked):
        rho = 0.985
        noise = (np.random.normal(size=(self.num_gnb_beams)) + 1j * np.random.normal(size=(self.num_gnb_beams))) * 0.1
        self.h_history = np.roll(self.h_history, -1, axis=0)
        self.h_history[-1] = rho * self.h_history[-2] + (1 - rho) * noise
        if is_blocked:
            self.h_history[-1] *= 0.0177
