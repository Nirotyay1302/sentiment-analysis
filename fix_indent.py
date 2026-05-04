with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('app.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if line.strip() == 'if y_pred_num is None:':
            f.write('                            if y_pred_num is None:\\n')
        elif line.strip() == 'st.markdown("### Prediction Previews")':
            f.write('                            st.markdown("### Prediction Previews")\\n')
        else:
            f.write(line)
