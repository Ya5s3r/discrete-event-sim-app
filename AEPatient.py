# class representing patients coming in
import random

class AEPatient:
    def __init__(self, p_id) -> None:
        self.p_id = p_id
        self.time_in_system = 0
        self.admitted = False

    def set_priority(self):
        # set priority according to weighted random choices - most are moderate in priority
        self.priority = random.choices([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1])[0]

    def set_triage_outcome(self):
        # decision tree - if priority 5, go to Minor Injury Unit (MIU) or home. Higher priority go to AE
        if self.priority <5:
            self.triage_outcome = 'AE'
        elif self.priority == 5: # of those who are priority 5, 20% will go home with advice, 80% go to 'MIU'
            self.triage_outcome = random.choices(['home', 'MIU'], [0.2, 0.8])[0]