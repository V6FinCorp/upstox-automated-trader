from .auth import broker_from_env

if __name__ == "__main__":
    broker = broker_from_env()
    # Place a test order (use a small quantity and a safe symbol)
    result = broker.place_order(
        instrument="NSE_EQ|INE002A01018",  # Try full instrument_key for Reliance
        quantity=1,
        side="BUY",
        order_type="MARKET",
        product_type="CNC"
    )
    print("Order result:", result)
