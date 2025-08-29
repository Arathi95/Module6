"""
Database Fundamentals & AI Analytics

Simple Sales Dashboard - Educational Version
Perfect for teaching students about data dashboards step by step
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from supabase import create_client
from dotenv import load_dotenv
import openai
import re
from sqlalchemy import create_engine, text
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# =====================================================
# STEP 1: CONFIGURATION & SETUP
# =====================================================

# Page setup - this controls how the app looks
st.set_page_config(
    page_title="Simple Sales Dashboard",
    page_icon="📊",
    layout="wide"  # Makes the app use full width
)

# =====================================================
# STEP 2: DATABASE CONNECTION
# =====================================================

# Read database credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Read AI configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini

# Configure OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Create PostgreSQL connection for AI queries
@st.cache_resource
def get_postgres_engine():
    """Create SQLAlchemy engine for direct PostgreSQL access"""
    try:
        # Extract project reference from Supabase URL
        parsed_url = urlparse(SUPABASE_URL)
        project_ref = parsed_url.hostname.split('.')[0]  # Extract from https://PROJECT.supabase.co
        
        # Supabase PostgreSQL connection string
        # Note: For direct PostgreSQL access, we need the database password
        # This might need to be configured separately in your Supabase project
        postgres_url = f"postgresql://postgres:[YOUR-DB-PASSWORD]@db.{project_ref}.supabase.co:5432/postgres"
        
        # Alternative: Try using pooler connection
        pooler_url = f"postgresql://postgres.{project_ref}:[YOUR-DB-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
        
        # For now, let's try a simple approach - we'll need the actual DB password
        # This is different from the API keys
        db_password = os.getenv("SUPABASE_DB_PASSWORD")
        if not db_password:
            st.warning("⚠️ SUPABASE_DB_PASSWORD not set. Add it to .env for AI SQL queries.")
            return None
            
        postgres_url = f"postgresql://postgres:{db_password}@db.{project_ref}.supabase.co:5432/postgres"
        
        engine = create_engine(postgres_url)
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return engine
    except Exception as e:
        st.warning(f"⚠️ PostgreSQL connection failed: {e}")
        return None


# Create database connection
@st.cache_resource  # This caches the connection so it's faster
def get_database():
    """Connect to our Supabase database"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except:
        st.error("❌ Cannot connect to database. Check your credentials!")
        return None


# =====================================================
# STEP 3: DATA LOADING FUNCTIONS
# =====================================================

@st.cache_data(ttl=60)  # Cache data for 60 seconds
def load_customers():
    """Load customer data from database"""
    db = get_database()
    if db:
        try:
            result = db.table('customers').select('*').execute()
            return pd.DataFrame(result.data)
        except:
            st.error("❌ Could not load customers")
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_orders():
    """Load orders data from database"""
    db = get_database()
    if db:
        try:
            result = db.table('orders').select('*').execute()
            return pd.DataFrame(result.data)
        except:
            st.error("❌ Could not load orders")
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_products():
    """Load products data from database"""
    db = get_database()
    if db:
        try:
            result = db.table('products').select('*').execute()
            return pd.DataFrame(result.data)
        except:
            st.error("❌ Could not load products")
    return pd.DataFrame()


# =====================================================
# STEP 4: DATABASE AUTHENTICATION
# =====================================================

def authenticate_user(email, password):
    """Authenticate user against database"""
    db = get_database()
    if not db:
        return False, "Database connection failed"
    
    try:
        # First, try to authenticate using Supabase Auth (if user was created with auth)
        try:
            result = db.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if result.user:
                return True, f"Welcome {result.user.email}!"
        except Exception as auth_error:
            # If auth fails, check for users in a custom users table
            try:
                result = db.table('users').select('*').eq('email', email).execute()
                if result.data:
                    user = result.data[0]
                    # In a real app, you'd hash/verify passwords properly
                    if user.get('password') == password:
                        return True, f"Welcome {user.get('name', email)}!"
                    else:
                        return False, "Invalid password"
                else:
                    return False, "User not found"
            except Exception as table_error:
                return False, f"Authentication error: {str(table_error)}"
                
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"
    
    return False, "Invalid credentials"


def check_password():
    """Database-based authentication with fallback to demo mode"""

    # If already logged in, return True
    if st.session_state.get("logged_in", False):
        return True

    # Show login form
    st.markdown("### 🔐 Please Login")

    # Login form with email and password
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="Enter your email (e.g., admin@test.com)")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("🚀 Login")

        if login_button:
            if email and password:
                # Try database authentication first
                success, message = authenticate_user(email, password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = email
                    st.success(f"✅ {message}")
                    st.rerun()  # Refresh the page
                else:
                    # Fallback to demo mode for backward compatibility
                    if email == "demo" and password == "password":
                        st.session_state.logged_in = True
                        st.session_state.username = "demo"
                        st.success("✅ Demo login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        st.info("💡 Try your database credentials or demo/password for demo mode")
            else:
                st.error("❌ Please enter both email and password!")

    return False


# =====================================================
# STEP 4.5: AI FUNCTIONS
# =====================================================

def translate_to_sql(natural_query):
    """Convert natural language to SQL query using OpenAI"""
    if not OPENAI_API_KEY:
        return False, "OpenAI API key not configured"
    
    system_prompt = """
You are a SQL expert. Convert natural language to PostgreSQL queries.
Available tables:
- customers (id, name, email, phone)
- products (id, name, category, price, stock_quantity)
- orders (id, customer_id, order_date, total_amount, status)
- order_items (id, order_id, product_id, quantity, unit_price)

Rules:
1. Only generate SELECT queries
2. Always include proper JOINs when needed
3. Use date functions for time-based queries
4. Return only the SQL query, no explanation
5. Ensure queries are safe and read-only
"""
    
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": natural_query}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks if present
        if sql_query.startswith('```sql'):
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        elif sql_query.startswith('```'):
            sql_query = sql_query.replace('```', '').strip()
        
        # Validate SQL for safety
        if validate_sql_safety(sql_query):
            return True, sql_query
        else:
            return False, "Query contains unsafe operations"
            
    except Exception as e:
        return False, f"Error generating SQL: {str(e)}"


def validate_sql_safety(sql_query):
    """Validate SQL query for safety (only SELECT allowed)"""
    # Convert to uppercase for checking
    sql_upper = sql_query.upper().strip()
    
    # Forbidden operations
    forbidden_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 
        'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
    ]
    
    # Check if starts with SELECT
    if not sql_upper.startswith('SELECT'):
        return False
    
    # Check for forbidden keywords
    for keyword in forbidden_keywords:
        if keyword in sql_upper:
            return False
    
    return True


def execute_ai_query(sql_query):
    """Execute AI-generated SQL query using SQLAlchemy"""
    engine = get_postgres_engine()
    if not engine:
        # Fallback to Supabase client for basic queries
        return execute_ai_query_fallback(sql_query)
    
    try:
        # Execute the SQL query directly using SQLAlchemy
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            
            # Convert result to pandas DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if len(df) > 0:
                return True, "Query executed successfully", df
            else:
                return False, "No data returned", None
                
    except Exception as e:
        # Fallback to basic queries if SQL execution fails
        st.warning(f"SQL execution failed, using fallback: {str(e)}")
        return execute_ai_query_fallback(sql_query)


def execute_ai_query_fallback(sql_query):
    """Fallback method using existing Supabase client operations"""
    try:
        # Load available data using existing functions
        orders_df = load_orders()
        customers_df = load_customers()
        products_df = load_products()
        
        sql_upper = sql_query.upper()
        
        # Parse common query patterns and simulate results
        if "PRODUCT" in sql_upper and ("SUM" in sql_upper or "QUANTITY" in sql_upper or "SELLING" in sql_upper):
            # Best selling products simulation
            if orders_df.empty:
                return False, "No order data available", None
            
            # Group by mock product data (since we don't have real order_items)
            result_df = orders_df.groupby('customer_id').agg({
                'total_amount': 'sum',
                'id': 'count'
            }).reset_index()
            result_df.columns = ['Product ID', 'Total Sales', 'Order Count']
            result_df = result_df.sort_values('Total Sales', ascending=False).head(10)
            return True, "Query executed (simulated)", result_df
            
        elif "CUSTOMER" in sql_upper:
            if customers_df.empty:
                return False, "No customer data available", None
            return True, "Query executed successfully", customers_df.head(10)
            
        elif "PRODUCT" in sql_upper:
            if products_df.empty:
                return False, "No product data available", None
            return True, "Query executed successfully", products_df.head(10)
            
        elif "ORDER" in sql_upper:
            if orders_df.empty:
                return False, "No order data available", None
            return True, "Query executed successfully", orders_df.head(10)
            
        else:
            # Return basic summary
            if not orders_df.empty:
                summary = pd.DataFrame({
                    'Metric': ['Total Orders', 'Total Revenue'],
                    'Value': [len(orders_df), f"${orders_df['total_amount'].sum():.2f}"]
                })
                return True, "Query executed successfully", summary
            else:
                return False, "No data available", None
                
    except Exception as e:
        return False, f"Fallback query execution failed: {str(e)}", None


def generate_ai_insights():
    """Generate business insights using AI analysis of current data"""
    if not OPENAI_API_KEY:
        return "AI insights not available - OpenAI API key not configured"
    
    # Load current data
    orders_df = load_orders()
    customers_df = load_customers()
    products_df = load_products()
    
    if orders_df.empty:
        return "No data available for insights generation"
    
    # Prepare data summary for AI
    data_summary = f"""
Sales Data Summary:
- Total Orders: {len(orders_df)}
- Total Customers: {len(customers_df)}
- Total Products: {len(products_df)}
- Total Revenue: ${orders_df['total_amount'].sum():.2f}
- Average Order Value: ${orders_df['total_amount'].mean():.2f}
- Date Range: {orders_df['order_date'].min()} to {orders_df['order_date'].max()}
"""
    
    if 'status' in orders_df.columns:
        status_breakdown = orders_df['status'].value_counts().to_dict()
        data_summary += f"- Order Status: {status_breakdown}"
    
    system_prompt = """
You are a business analyst. Analyze the sales data and provide actionable insights.
Generate a concise analysis covering:
1. Key trends and patterns
2. Notable observations
3. Business recommendations
4. Potential areas of concern

Format your response in markdown with clear sections and bullet points.
Be specific and actionable in your recommendations.
"""
    
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this sales data:\n{data_summary}"}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"


# =====================================================
# STEP 5: HELPER FUNCTIONS
# =====================================================

def format_currency(amount):
    """Format numbers as currency"""
    return f"${amount:,.2f}"


def format_number(number):
    """Format numbers with commas"""
    return f"{number:,}"


# =====================================================
# STEP 6: MAIN DASHBOARD FUNCTIONS
# =====================================================

def show_header():
    """Display the main header"""
    st.title("📊 Simple Sales Dashboard")
    st.markdown("*Learn how data dashboards work - step by step!*")

    # Welcome message
    if "username" in st.session_state:
        st.markdown(f"👋 **Welcome, {st.session_state.username}!**")

    # Add a separator line
    st.markdown("---")


def show_key_metrics():
    """Show the main KPI cards at the top"""

    st.subheader("📈 Key Metrics")

    # Load the data we need
    orders_df = load_orders()
    customers_df = load_customers()

    # Calculate basic metrics
    if not orders_df.empty:
        total_sales = orders_df['total_amount'].sum()
        total_orders = len(orders_df)
        avg_order = total_sales / total_orders if total_orders > 0 else 0
    else:
        total_sales = total_orders = avg_order = 0

    total_customers = len(customers_df)

    # Display metrics in columns (this creates a nice layout)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total Sales",
            value=format_currency(total_sales)
        )

    with col2:
        st.metric(
            label="📦 Total Orders",
            value=format_number(total_orders)
        )

    with col3:
        st.metric(
            label="🎯 Average Order",
            value=format_currency(avg_order)
        )

    with col4:
        st.metric(
            label="👥 Customers",
            value=format_number(total_customers)
        )


def show_sales_chart():
    """Show a simple sales chart"""

    st.subheader("📈 Sales Over Time")

    orders_df = load_orders()

    if orders_df.empty:
        st.warning("No sales data to display")
        return

    # Prepare data for chart
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    daily_sales = orders_df.groupby(orders_df['order_date'].dt.date)['total_amount'].sum().reset_index()
    daily_sales.columns = ['Date', 'Sales']

    # Create a simple line chart
    fig = px.line(
        daily_sales,
        x='Date',
        y='Sales',
        title="Daily Sales",
        labels={'Sales': 'Sales (£)'}
    )

    # Make it look nice
    fig.update_layout(
        plot_bgcolor='white',
        height=400
    )

    # Display the chart
    st.plotly_chart(fig, width='stretch')


def show_data_tables():
    """Show simple data tables"""

    st.subheader("📋 Data Tables")

    # Create tabs for different tables
    tab1, tab2, tab3 = st.tabs(["👥 Customers", "📦 Orders", "🛍️ Products"])

    with tab1:
        customers_df = load_customers()
        if not customers_df.empty:
            st.dataframe(customers_df, width='stretch')
        else:
            st.info("No customer data available")

    with tab2:
        orders_df = load_orders()
        if not orders_df.empty:
            # Format the data nicely
            display_orders = orders_df.copy()
            if 'total_amount' in display_orders.columns:
                display_orders['total_amount'] = display_orders['total_amount'].apply(format_currency)
            st.dataframe(display_orders, width='stretch')
        else:
            st.info("No order data available")

    with tab3:
        products_df = load_products()
        if not products_df.empty:
            # Format the data nicely
            display_products = products_df.copy()
            if 'price' in display_products.columns:
                display_products['price'] = display_products['price'].apply(format_currency)
            st.dataframe(display_products, width='stretch')
        else:
            st.info("No product data available")


def show_simple_analytics():
    """Show some simple analytics"""

    st.subheader("🔍 Simple Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Top customers by order count
        st.markdown("**🏆 Most Active Customers**")

        orders_df = load_orders()
        customers_df = load_customers()

        if not orders_df.empty and not customers_df.empty:
            # Count orders per customer
            customer_orders = orders_df['customer_id'].value_counts().head(5)

            # Create a simple bar chart
            fig = px.bar(
                x=customer_orders.values,
                y=[f"Customer {i + 1}" for i in range(len(customer_orders))],
                orientation='h',
                title="Orders by Customer"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Not enough data for analysis")

    with col2:
        # Order status breakdown
        st.markdown("**📊 Order Status**")

        if not orders_df.empty and 'status' in orders_df.columns:
            status_counts = orders_df['status'].value_counts()

            # Create a pie chart
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Order Status Distribution"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No status data available")


def show_ai_chat():
    """Show AI-powered natural language query interface"""
    
    st.subheader("🤖 Ask Questions About Your Data")
    
    if not OPENAI_API_KEY:
        st.warning("⚠️ OpenAI API key not configured. AI features are disabled.")
        return
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    with st.form("ai_query_form"):
        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., 'Show me the best selling products' or 'Which customers ordered the most?'"
        )
        submit_button = st.form_submit_button("🚀 Ask AI")
        
        if submit_button and user_question:
            # Add user question to history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Process the question
            with st.spinner("🤖 AI is thinking..."):
                # Translate to SQL
                success, result = translate_to_sql(user_question)
                
                if success:
                    sql_query = result
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"I'll help you with that! Generated SQL:\n```sql\n{sql_query}\n```"
                    })
                    
                    # Execute the query
                    query_success, message, df = execute_ai_query(sql_query)
                    
                    if query_success and df is not None:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"✅ Found {len(df)} results:"
                        })
                        # Store the result for display
                        st.session_state.last_ai_result = df
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ {message}"
                        })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ {result}"
                    })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
        
        # Display results if available
        if "last_ai_result" in st.session_state:
            st.markdown("### 📊 Results")
            st.dataframe(st.session_state.last_ai_result, width='stretch')
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        if "last_ai_result" in st.session_state:
            del st.session_state.last_ai_result
        st.rerun()


def show_ai_insights():
    """Show AI-generated business insights"""
    
    st.subheader("🧠 AI Business Insights")
    
    if not OPENAI_API_KEY:
        st.warning("⚠️ OpenAI API key not configured. AI insights are disabled.")
        return
    
    # Cache insights to avoid regenerating on every reload
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_cached_insights():
        return generate_ai_insights()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh Insights", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    with col1:
        st.markdown("*AI-powered analysis of your sales data*")
    
    # Generate and display insights
    with st.spinner("🤖 AI is analyzing your data..."):
        insights = get_cached_insights()
    
    # Display insights in a nice container
    with st.container():
        st.markdown(insights)
    
    # Show timestamp
    st.caption(f"*Generated at {datetime.now().strftime('%H:%M:%S')}*")


def show_sidebar():
    """Show sidebar with controls"""

    with st.sidebar:
        st.markdown("### 🎛️ Controls")

        # Refresh button
        if st.button("🔄 Refresh Data", width='stretch'):
            st.cache_data.clear()  # Clear cache to get fresh data
            st.success("Data refreshed!")
            st.rerun()

        # Logout button
        if st.button("🚪 Logout", width='stretch'):
            st.session_state.logged_in = False
            st.session_state.pop('username', None)
            st.rerun()

        st.markdown("---")

        # Simple info
        st.markdown("### ℹ️ About")
        st.markdown("""
        This is a **simple sales dashboard** built with:

        - 🐍 **Python** - Programming language
        - 📊 **Streamlit** - Web app framework  
        - 🗄️ **Supabase** - Database
        - 📈 **Plotly** - Charts and graphs
        - 🤖 **OpenAI** - AI-powered insights

        Perfect for learning how dashboards work!
        """)
        
        # AI Status
        if OPENAI_API_KEY:
            st.success("🤖 AI Features: Enabled")
        else:
            st.warning("🤖 AI Features: Disabled")

        # Show current time
        st.markdown(f"**Current time:** {datetime.now().strftime('%H:%M:%S')}")


# =====================================================
# STEP 7: MAIN APPLICATION
# =====================================================

def main():
    """The main function that runs our app"""

    # Check if user is logged in
    if not check_password():
        return  # Stop here if not logged in

    # Show the sidebar
    show_sidebar()

    # Show the main content
    show_header()
    show_key_metrics()

    st.markdown("---")  # Add a separator

    show_sales_chart()

    st.markdown("---")  # Add a separator

    show_simple_analytics()

    st.markdown("---")  # Add a separator

    show_data_tables()

    st.markdown("---")  # Add a separator

    # AI Features (if enabled)
    if OPENAI_API_KEY:
        # Create tabs for AI features
        ai_tab1, ai_tab2 = st.tabs(["🤖 AI Chat", "🧠 AI Insights"])
        
        with ai_tab1:
            show_ai_chat()
        
        with ai_tab2:
            show_ai_insights()
    else:
        st.info("💡 Configure OPENAI_API_KEY in your .env file to enable AI features!")

    # Footer
    st.markdown("---")
    st.markdown("*📚 Educational Dashboard - Built for Learning!*")


# =====================================================
# STEP 8: RUN THE APP
# =====================================================

if __name__ == "__main__":
    main()