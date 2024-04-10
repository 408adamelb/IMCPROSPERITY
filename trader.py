from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

import pandas as pd
import matplotlib.pyplot as plt
from wavelet import WaveletTransform, getExponent
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score




class Trader:
    
    def predict_next_price(self, prices):
        transform = WaveletTransform(waveletName="db4")
        coefficients = transform.dwt(prices, level=6)
        reconstructed_signal = np.array(transform.idwt(coefficients, level=6))
        return reconstructed_signal
        
    
    def run(self, state: TradingState):
        result = {}
        orders: List[Order] = []
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            # mid_price = (best_ask + best_bid) / 2
            
        if (state.timestamp > 1000):
            prices = state.traderData.split(" ")[:-1]
            asks = []
            bids = []
            
            for index, price in enumerate(prices):
                if index % 2 == 0:
                    asks.append(float(price))
                else:
                    bids.append(float(price))
                    
            next_ask = self.predict_next_price(asks[-6:])
            next_bid = self.predict_next_price(bids[-6:])
            
            best_bid_idx = np.argmax(np.array(next_bid))
            best_ask_idx = np.argmin(np.array(next_ask))
            
            # if (best_bid_idx < best_ask_idx):
            print("BUY", str(-best_ask_amount) + "x", next_ask[best_ask_idx])
            orders.append(Order(product, next_ask[best_ask_idx], -best_ask_amount))
            # else:
            
            # if (best)
            print("SELL", str(best_bid_amount) + "x", next_bid[best_bid_idx])
            orders.append(Order(product, next_bid[best_bid_idx], best_bid_amount))
            for product2 in state.order_depths:
                order_depth: OrderDepth = state.order_depths[product]
            
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                mid = (best_ask  + best_bid )/2

            if (state.timestamp > 1000):
                prices = state.traderData.split(" ")[:-1]
                asks = np.array([])
                asks.append(best_ask)
                bids = np.array([])
                bids.append(best_bid)
                mids = np.arra([])
            
                for index, price in enumerate(prices):
                    if index % 2 == 0:
                        asks.append(float(price))

                    else:
                        bids.append(float(price))
                mids = asks+bids /2
                if(mid[state.timestamp] + mids[state.timestamp-1])/2 > mids[state.timestamp-1]:

                    orders.append(Order(product, mids[state.timestamp-1] +2,-3 ))
                    orders.append(Order(product, mids[state.timestamp-1] +1,-2 ))
                    orders.append(Order(product, mids[state.timestamp-1] +2,-1 ))
                else:
                    orders.append(Order(product, mids[state.timestamp-1] -2,3 ))
                    orders.append(Order(product, mids[state.timestamp-1] -1,2 ))
                    orders.append(Order(product, mids[state.timestamp-1] ,1 ))

            # mid_price = (best_ask + best_bid) / 2
            
        # if (state.timestamp > 1000):
        #     for index, price in enumerate(prices):
        #         if index % 2 == 0:
        #             asks.append(float(price))
        #         else:
        #             bids.append(float(price))
                    
        #     next_ask = self.predict_next_price(asks[-10:])
        #     next_bid = self.predict_next_price(bids[-10:])
            
            
        #     best_bid_idx = np.argmax(np.array(next_bid))
        #     best_ask_idx = np.argmin(np.array(next_ask))
            
        #     if (best_bid_idx > best_ask_idx):
        #         orders.append(Order(product, next_ask[best_ask_idx], -best_ask_amount))
        #     else:
        #         orders.append(Order(product, next_bid[best_bid_idx], best_bid_amount))
                
            
            
            
            # print(next_ask[best_ask_idx], next_bid[best_bid_idx])
            # print(next_ask[best_ask_idx])
            
            
            
            
            
            
            
                # print((best_ask + best_bid) / 2)
            
            
            
            
        #     orders: List[Order] = []
        #     acceptable_price = 10;  # Participant should calculate this value
        #     print("Acceptable price : " + str(acceptable_price))
        #     print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
        #     if len(order_depth.sell_orders) != 0:
        #         best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        #         if int(best_ask) < acceptable_price:
        #             print("BUY", str(-best_ask_amount) + "x", best_ask)
        #             orders.append(Order(product, best_ask, -best_ask_amount))
    
        #     if len(order_depth.buy_orders) != 0:
        #         best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        #         if int(best_bid) > acceptable_price:
        #             print("SELL", str(best_bid_amount) + "x", best_bid)
        #             orders.append(Order(product, best_bid, -best_bid_amount))
            
        result[product] = orders
    
        traderData = state.traderData + f"{best_ask} " + f"{best_bid} "
        
        conversions = 1
        return result, conversions, traderData
