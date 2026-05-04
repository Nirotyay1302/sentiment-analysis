import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if 'em_counts.columns = ["emotion", "count"]' in line:
        # Check if the next line is "if not extracted_texts:"
        if i + 1 < len(lines) and 'if not extracted_texts:' in lines[i+1]:
            start_idx = i
            break

end_idx = -1
for i in range(start_idx, len(lines)):
    if '# ----------- Mode 7: Train Custom Model -----------' in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    # We replace from start_idx + 1 up to end_idx with the proper completion of Mode 6
    replacement = [
        "                        fig_em = px.pie(em_counts, values='count', names='emotion', hole=0.3, title='Emotion Distribution', color_discrete_sequence=px.colors.qualitative.Pastel)\n",
        "                        st.plotly_chart(fig_em, use_container_width=True)\n",
        "        else:\n",
        "            st.error(f\"Failed to fetch history: {response.text}\")\n",
        "    except Exception as e:\n",
        "        st.error(f\"Database unreachable. Please ensure FastAPI server is running. Error: {e}\")\n"
    ]
    
    new_lines = lines[:start_idx + 1] + replacement + lines[end_idx:]
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Fix applied successfully.")
else:
    print(f"Could not find indices. start_idx: {start_idx}, end_idx: {end_idx}")
