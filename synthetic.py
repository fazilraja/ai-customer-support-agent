from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta, date
import json
import os
from mirascope import llm
from mirascope.core.google import google_call
from dotenv import load_dotenv
from openai import OpenAI

    
# Load environment variables from .env file
load_dotenv()
class ShippingAddress(BaseModel):
    street: str
    city: str
    state: str
    zip: str

class Preferences(BaseModel):
    payment: str
    delivery: str
    communication: str

class CustomerProfile(BaseModel):
    customerId: str
    name: str
    email: str
    phone: str
    dob: str
    shippingAddress: ShippingAddress
    preferences: Preferences

class Product(BaseModel):
    productName: str
    price: float
    quantity: int

class Order(BaseModel):
    orderId: str
    orderDate: str
    deliveryDate: str
    products: List[Product]
    status: str

class CustomerServiceInteraction(BaseModel):
    interactionId: str
    orderId: str
    date: str
    description: str

class CustomerData(BaseModel):
    customerProfile: CustomerProfile
    orderHistory: List[Order]
    customerServiceInteractions: List[CustomerServiceInteraction]
    
    
def generate_synthetic_data_openai(user_id: Optional[str] = None):
    """Generate synthetic customer data for a given user."""
    today = datetime.now()
    order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
    expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")
    
    # Set the API key directly
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""Generate a detailed customer profile and order history for a TechGadgets.com customer with ID {user_id}.
        When generating the recent order, make it a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})"""
                    
    customer_data = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a data generation AI that creates realistic customer profiles and order hitories"},
                {"role": "user", "content": prompt}
            ],
            response_format=CustomerData
        )
            
    # First convert the Pydantic model to a dict
    customer_data_dict = customer_data.model_dump()
    
    # Then convert to JSON string
    customer_data_json = json.dumps(customer_data_dict)
    
    # Write to file for debugging
    write_customer_data_to_file(customer_data_json, user_id)
    
def generate_synthetic_data_mirascope_gemini(user_id: Optional[str] = None):
    """Generate synthetic customer data for a given user using Mirascope framework.
    """

    customer_data = generate_data_mirascope(user_id)
            
    # First convert the Pydantic model to a dict
    customer_data_dict = customer_data.model_dump()
    
    # Then convert to JSON string
    customer_data_json = json.dumps(customer_data_dict)
    
    # Write to file for debugging
    write_customer_data_to_file(customer_data_json, user_id)
    
@google_call(model="gemini-2.0-flash", response_model=CustomerData)
def generate_data_mirascope(user_id: Optional[str] = None):
    today = datetime.now()
    order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
    expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")
    
    return f"""Generate a detailed customer profile and order history for a TechGadgets.com customer with ID {user_id}.
        When generating the recent order, make it a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})"""
    
def generate_synthetic_data_gemini(user_id: Optional[str] = None) -> CustomerData:
    """Generate synthetic customer data for a given user using Google's Generative AI SDK.
    
    Args:
        user_id: Optional customer ID to use in the generated data.
        
    Returns:
        CustomerData: Generated customer profile and history.
        
    Raises:
        ValueError: If GENAI_API_KEY environment variable is not set
    """
    from google import genai
    import os
    
    # Get API key with better error handling
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "GENAI_API_KEY environment variable not found. Please ensure it is set in your environment "
            "or .env file. You can set it by:\n"
            "1. Adding GENAI_API_KEY=your_key to .env file, or\n"
            "2. Running: export GENAI_API_KEY=your_key in your terminal"
        )
    
    # Configure the client
    client = genai.Client(api_key=api_key)
    
    today = datetime.now()
    order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
    expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")
    
    prompt = f"""Generate a detailed customer profile and order history for a TechGadgets.com customer with ID {user_id}.
        When generating the recent order, make it a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})."""
    
    # Generate the response with type-safe parsing
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': CustomerData,
        },
    )
    
    # Get the parsed response
    customer_data: CustomerData = response.parsed
            
    # Convert to JSON and write to file for debugging
    customer_data_json = customer_data.model_dump_json()
    write_customer_data_to_file(customer_data_json, user_id)
    
    return customer_data
    
def write_customer_data_to_file(customer_data_json: str, user_id: str):
        """Write customer data JSON to a file for debugging/readability.
        
        Args:
            customer_data_json: The JSON string to write
            user_id: The user ID to include in the filename
        """
        import json
        from datetime import datetime
        
        # Create a debug directory if it doesn't exist
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{debug_dir}/customer_data_{user_id}_{timestamp}.json"
        
        # Parse and re-write the JSON string to get pretty formatting
        parsed_json = json.loads(customer_data_json)
        
        # Write the data with nice formatting
        with open(filename, 'w') as f:
            json.dump(
                parsed_json,
                f,
                indent=2,  # Pretty print with 2-space indentation
                default=str  # Handle date objects
            )
        
        print(f"Wrote customer data JSON to {filename}")
        
generate_synthetic_data_mirascope_gemini(user_id="123")
generate_synthetic_data_gemini(user_id="123")
generate_synthetic_data_openai(user_id="123")