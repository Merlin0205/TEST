import streamlit as st

def main():
    """
    Main function to display text and a button using Streamlit.
    This function writes "Ahoj svete" to the Streamlit app and creates a button labeled "Klikni na mě".
    When the button is clicked, it writes "Tlačítko bylo stisknuto!" to the app.
    """
    st.write("Ahoj svete")  # Displays text
    
    if st.button("Klikni na mě"):  # Creates a button
        st.write("Tlačítko bylo stisknuto!")

if __name__ == "__main__":
    main()