import html
import pickle

from .image_utils import convert_pil_image_to_base64

def encode_html_str(s):
    return html.encode(s).replace("\n", "<br>")

class CustomLogger:
    def __init__(self):
        self.logs = []

    def log(self, name=None, log_type=None, message=None, image=None, content=None):
        self.logs.append({
                'name': name, 
                'type': log_type, 
                'message': message, 
                'image': image,
                'content': content
            }
        )

    def get_logs(self):
        return self.logs
    
    def clear(self):
        self.logs.clear()

    def output_logs_to_notebook(self):
        """
        Outputs the log history to the notebook, mimicking Python's default logger format.
        Renders images if present.
        """
        try:
            from IPython.display import display
        except:
            print("Faile to import display from IPython.display. Are you in a notebook env?")
            return
        
        for log_entry in self.logs:
            log_output = f"{log_entry['name']} - {log_entry['type'].upper()} - {log_entry['message']}"
            print(log_output)
            if log_entry['image'] is not None:
                display(log_entry['image'])


    def save_logs(self, filename):
        """
        Saves the logs to a file, including all content such as ndarray, torch.tensor, and images.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.logs, file)


    def logs_to_html(self):
        """
        Converts the logs into a styled HTML string that mimics a chatbot UI,
        with alignment based on log type and handling both message and image fields.
        """
        html_string = '<div style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px; margin: 20px auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">'
        for log in self.logs:
            # The chat box's position
            align = 'right' if log['type'] == 'call' else 'left'
            background_color = '#f1f0f0' if align == 'left' else '#d1e7dd'
            border_color = '#ccc' if align == 'left' else '#a6d1b2'
            html_string += f'''
                <div style="text-align: {align}; margin-bottom: 10px;">
                    <div style="display: inline-block; background-color: {background_color}; padding: 10px; border-radius: 10px; border: 2px solid {border_color};">
                        <b>{log["name"]}:</b><br/>
            '''
            if log['message']:
                message = encode_html_str(log['message'])
                html_string += f'<div>{message}</div>'
            if log['image'] is not None:
                image_base64 = convert_pil_image_to_base64(log['image'])
                img_html = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 100%; height: auto;"/>'
                html_string += img_html
            if log['content'] is not None:
                html_string += '<div>This message contains binary content. Please view the pkl file.</div>'
            html_string += '</div></div>'
        html_string += '</div>'
        return html_string
    

    def save_logs_to_html_file(self, file_path):
        """
        Saves the HTML logs to a specified file path.
        """
        html_logs = self.logs_to_html()
        with open(file_path, 'w') as file:
            file.write(html_logs)


    def display_html_logs_in_notebook(self):
        """
        Displays the HTML logs in a Jupyter Notebook.
        """
        from IPython.display import HTML
        html_logs = self.logs_to_html()
        return HTML(html_logs)

    
    
# class LogWrapper:
#     def __init__(self, wrapped, logger):
#         self._wrapped = wrapped
#         self.logger = logger

#     def __getattr__(self, name):
#         orig_attr = getattr(self._wrapped, name)
#         if callable(orig_attr):
#             def wrapper(*args, **kwargs):
#                 self.logger.log(name, "call", {"args": args, "kwargs": kwargs})
#                 try:
#                     result = orig_attr(*args, **kwargs)
#                     self.logger.log(name, "return", {"result": result})
#                     return result
#                 except Exception as e:
#                     # This will be triggered only by internal errors of the API calls. 
#                     # Custom expections are raised in `agent.plan()`` and will not be caught here.
#                     self.logger.log(name, "unexpected_exception", {"exception": str(e)})
#                     raise
#             return wrapper
#         return orig_attr
    