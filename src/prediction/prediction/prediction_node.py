import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd

class product():
    def __init__(self, name, probability, coordinates):
        self.name = name
        self.probability = probability
        self.coordinates = coordinates

# class item_relations():
#     def __init__(self, items):
#         self.items = items
#         self.relations = pd.DataFrame(index=items, columns=items, data=pd.NA)

#     def add_relation(self, item2, item2, score):
#         if item1 in self.items and item2 in self.items:
#             self.relations.at[item1, item2] = score
#             self.relations.at[item2, item1] = score

#     def get_relation(self, item1, item2):
#         if item1 in self.items and item2 in self.items:
#             return self.relations.at[item1, item2]
#         else:
#             return None

class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction_node')    
        
        self.items = [] # list of item objects

        self.items.append(product('Beverage', 0, 0))
        self.items.append(product('Canned Goods', 0, 0))
        self.items.append(product('Dishwashing Liquid', 0, 0))
        self.items.append(product('Biscuits', 0, 0))
        self.items.append(product('Tissue', 0, 0))

        # Defining item relations (co-occurence score)
        score = {
            'Beverage': {'Canned Goods': 0, 'Dishwashing Liquid': 0, 'Biscuits': 0, 'Tissue': 0},
            'Canned Goods': {'Beverage': 0, 'Dishwashing Liquid': 0, 'Biscuits': 0, 'Tissue': 0},
            'Dishwashing Liquid': {'Beverage': 0, 'Canned Goods': 0, 'Biscuits': 0, 'Tissue': 0},
            'Biscuits': {'Beverage': 0, 'Canned Goods': 0, 'Dishwashing Liquid': 0, 'Tissue': 0},
            'Tissue': {'Beverage': 0, 'Canned Goods': 0, 'Dishwashing Liquid': 0, 'Biscuits': 0}
        }

        # Converting score data to a DataFrame
        df = pd.DataFrame.from_dict(score, orient='index')
        np.fill_diagonal(df.values, 0)

        self.held_items = [] # list of held item objects
        
        # subscribe to the ff: held items, nav2 path distance, mediapipe pose angle deviation
        self.create_subscription(
            held_items,
            'held_items',
            self.held_items_callback,
            10
        )
        
        # publish the ff: velocity commands 

    def compute_probability(self):
        for item in self.items:
            prior = item.probability
            likelihood = held_items_likelihood(item)

    # def compute_evidence():
    #     for item in self.items:
    #         evidence = 0
    #         for other_item in self.items:

    def held_items_callback(self, msg):
        # Add item to the list
        name = msg.name
        self.held_items.append(name)

    def held_items_likelihood(self, item, alpha=1.0):
        # Compute the likelihood of the held items given the item
        if len(self.held_items) == 0:
            return 1.0

        scores = []
        for held_item in self.held_items:
            numerator = self.get_relation(item.name, held_item)
            denominator = sum(self.get_relation(item.name, other_item) for other_item in self.items if other_item != held_item)
            if denominator == 0:
                continue
            smoothed = (numerator + alpha) / (denominator + alpha * len(self.items))
            scores.append(smoothed)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
