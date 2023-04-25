import streamlit


streamlit.code(
    """


fn main() {

    let deez: i32 = 69;

    println!("deez nuts");
}



""",
    language="rust",
)


values = streamlit.slider(label="pick a number", min_value=0, max_value=100)

streamlit.write("Values:", values)
