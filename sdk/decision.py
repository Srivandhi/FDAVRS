class DecisionMaker:
    def __init__(self, high_drift_threshold=0.7, low_drift_threshold=0.3):
        self.high_thresh = high_drift_threshold
        self.low_thresh = low_drift_threshold

    def decide(self, metrics):
        score = metrics['score']

        if score > self.high_thresh:
            return "SERVER_UPDATE_REQUIRED"
        elif score > self.low_thresh:
            return "LOCAL_ADAPTATION"
        else:
            return "IDLE" # Restoring 'IDLE' label as per user's previous output