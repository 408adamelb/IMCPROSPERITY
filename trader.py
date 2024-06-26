from datamodel import OrderDepth, UserId, TradingState, Order
import collections
from typing import List
import numpy as np
import math
import string


class DB4WaveletTransform:
    """
    Class to run the selected wavelet and to perform the dwt & idwt
    based on the wavelet filters

    Attributes
    ----------
    __wavelet__: object
        object of the selected wavelet class
    """



    def __init__(self):
        self.__name__ = "Daubechies Wavelet 4"
        self.__motherWaveletLength__ = 8  # length of the mother wavelet
        self.__transformWaveletLength__ = 2  # minimum wavelength of input signal

        # decomposition filter
        # low-pass
        self.decompositionLowFilter = [
            -0.010597401784997278,
            0.032883011666982945,
            0.030841381835986965,
            - 0.18703481171888114,
            - 0.02798376941698385,
            0.6308807679295904,
            0.7148465705525415,
            0.23037781330885523
        ]

        # high-pass
        self.decompositionHighFilter = [
            -0.23037781330885523,
            0.7148465705525415,
            - 0.6308807679295904,
            - 0.02798376941698385,
            0.18703481171888114,
            0.030841381835986965,
            - 0.032883011666982945,
            - 0.010597401784997278,
        ]

        # reconstruction filters
        # low pass
        self.reconstructionLowFilter = [
            0.23037781330885523,
            0.7148465705525415,
            0.6308807679295904,
            - 0.02798376941698385,
            - 0.18703481171888114,
            0.030841381835986965,
            0.032883011666982945,
            - 0.010597401784997278,
        ]

        # high-pass
        self.reconstructionHighFilter = [
            -0.010597401784997278,
            - 0.032883011666982945,
            0.030841381835986965,
            0.18703481171888114,
            - 0.02798376941698385,
            - 0.6308807679295904,
            0.7148465705525415,
            - 0.23037781330885523,
        ]

    def dwt(self, arrTime, level):
        """
        Discrete Wavelet Transform

        Parameters
        ----------
        arrTime : array_like
            input array in Time domain
        level : int
            level to decompose

        Returns
        -------
        array_like
            output array in Frequency or the Hilbert domain
        """
        arrHilbert = [0.] * level
        # shrinking value 8 -> 4 -> 2
        a = level >> 1

        for i in range(a):
            for j in range(self.__motherWaveletLength__):
                k = (i << 1) + j

                # circulate the array if scale is higher
                while k >= level:
                    k -= level

                # approx & detail coefficient
                arrHilbert[i] += arrTime[k] * self.decompositionLowFilter[j]
                arrHilbert[i + a] += arrTime[k] * self.decompositionHighFilter[j]

        return arrHilbert

    def idwt(self, arrHilbert, level):
        """
        Inverse Discrete Wavelet Transform

        Parameters
        ----------
        arrHilbert : array_like
            input array in Frequency or the Hilbert domain
        level : int
            level to decompose

        Returns
        -------
        array_like
            output array in Time domain
        """
        arrTime = [0.] * level
        # shrinking value 8 -> 4 -> 2
        a = level >> 1

        for i in range(a):
            for j in range(self.__motherWaveletLength__):
                k = (i << 1) + j

                # circulating the array if scale is higher
                while k >= level:
                    k -= level

                # summing the approx & detail coefficient
                arrTime[k] += (arrHilbert[i] * self.reconstructionLowFilter[j] +
                               arrHilbert[i + a] * self.reconstructionHighFilter[j])

        return arrTime

class Trader:
    
    def predict_next_price(self, prices, level):
        transform = DB4WaveletTransform()
        coefficients = transform.dwt(prices, level=level)
        reconstructed_signal = np.array(transform.idwt(coefficients, level=level))
        return reconstructed_signal
    
    # COPIED FROM STANFORD
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return best_val
    
    def run(self, state: TradingState):
        result = {}
        
        tot_ask_vol = {}
        tot_bid_vol = {}
        
        tot_ask_price_vol = {}
        tot_bid_price_vol = {}
        
        next_ask_prices = {}
        next_bid_prices = {}
        
        next_mid_prices = {}
        cur_mid_prices = {}
        
        sell_orders = {}
        buy_orders = {}
        
        best_buy = {}
        best_sell = {}
        
        for product in state.order_depths:
            tot_ask_vol[product] = 0
            tot_bid_vol[product] = 0
            
            tot_ask_price_vol[product] = 0
            tot_bid_price_vol[product] = 0
            
            next_ask_prices[product] = 0
            next_bid_prices[product] = 0
            
            next_mid_prices[product] = 0
            cur_mid_prices[product] = 0
            
            order_depth: OrderDepth = state.order_depths[product]
            sell_orders[product] = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            buy_orders[product] = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
            
            for ask_price, ask_vol in list(sell_orders[product].items()):
                tot_ask_price_vol[product] += (ask_price * -ask_vol)
                tot_ask_vol[product] += -ask_vol
            
            for bid_price, bid_vol in list(buy_orders[product].items()):
                tot_bid_price_vol[product] += (bid_price * bid_vol)
                tot_bid_vol[product] += bid_vol
                
            next_bid_prices[product] = tot_bid_price_vol[product]/tot_bid_vol[product]
            next_ask_prices[product] = tot_ask_price_vol[product]/tot_ask_vol[product]
            
            next_mid_prices[product] = (next_ask_prices[product] + next_bid_prices[product])/2
            cur_mid_prices[product] = (next(iter(sell_orders[product])) + next(iter(buy_orders[product])))/2
            
        for product in next_mid_prices:
            orders: List[Order] = []
            if (product == "AMETHYSTS"):
                acc_price = 10000
            else:
                acc_price = int(next_mid_prices[product])
            
            cur_pos = 0
            CUR_POS_STATIC = 0
            if product in state.position:
                CUR_POS_STATIC = state.position[product]
                cur_pos = state.position[product]
            
            for ask_price, ask_vol in list(sell_orders[product].items()):
                if ((ask_price < acc_price) or ((CUR_POS_STATIC < 0) and (ask_price == acc_price) and cur_pos < 20)):
                    buy_vol = min(-ask_vol, 20-cur_pos)
                    cur_pos += buy_vol
                    orders.append(Order(product, ask_price, buy_vol))
                    
            best_sell = self.values_extract(sell_orders[product])
            best_buy = self.values_extract(buy_orders[product], 1)
            
            undercut_buy = best_buy + 1
            undercut_sell = best_sell - 1
            
            our_bid = min(undercut_buy, acc_price-1)
            our_ask = min(undercut_sell,acc_price+1)
            
            if (cur_pos < 20) and (CUR_POS_STATIC < 0):
                num = min(40, 20 - cur_pos)
                orders.append(Order(product, min(undercut_buy+1, acc_price-1), num))
                cur_pos += num

            if (cur_pos < 20) and (CUR_POS_STATIC > 15):
                num = min(40, 20 - cur_pos)
                orders.append(Order(product, min(undercut_buy-1, acc_price-1), num))
                cur_pos += num

            if cur_pos < 20:
                num = min(40, 20 - cur_pos)
                orders.append(Order(product, our_bid, num))
                cur_pos += num
            
            cur_pos = CUR_POS_STATIC
            
            for bid_price, bid_vol in list(buy_orders[product].items()):
                if ((bid_price > acc_price) or ((CUR_POS_STATIC > 0) and (bid_price == acc_price) and cur_pos > -20)):
                    sell_vol = max(-bid_vol, -20-cur_pos)
                    cur_pos += sell_vol
                    orders.append(Order(product, bid_price, sell_vol))
                    
            if (cur_pos > -20) and (CUR_POS_STATIC > 0):
                num = max(-40, -20-cur_pos)
                orders.append(Order(product, max(undercut_sell-1, acc_price+1), num))
                cur_pos += num

            if (cur_pos > -20) and (CUR_POS_STATIC < -15):
                num = max(-40, -20-cur_pos)
                orders.append(Order(product, max(undercut_sell+1, acc_price+1), num))
                cur_pos += num

            if cur_pos > -20:
                num = max(-40, -20-cur_pos)
                orders.append(Order(product, our_ask, num))
                cur_pos += num
            
            result[product] = orders
        
        traderData = state.traderData
        conversions = 1
        
        return result, conversions, traderData
