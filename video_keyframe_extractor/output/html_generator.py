from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import shutil

class HTMLGenerator:
    def __init__(self, template_dir="video_keyframe_extractor/output/templates", output_dir="output_documents"):
        self.output_dir = output_dir
        self.template_dir = template_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Jinja2 env
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate_document(self, data: dict, project_name: str) -> str:
        """
        Generates the final HTML document.
        data: Dict containing 'results' (list of segments with text/keyframe).
        """
        print(f"Generating HTML for project: {project_name}")
        
        project_dir = os.path.join(self.output_dir, project_name)
        assets_dir = os.path.join(project_dir, "assets")
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(assets_dir, exist_ok=True)
        
        # Copy selected images to assets
        segments = data.get("segments", [])
        for i, seg in enumerate(segments):
            kf = seg.get("keyframe")
            if kf and kf.get("path"):
                src_path = kf["path"]
                if os.path.exists(src_path):
                    # Rename for cleanliness: frame_001.jpg
                    ext = os.path.splitext(src_path)[1]
                    dst_filename = f"frame_{i:03d}{ext}"
                    dst_path = os.path.join(assets_dir, dst_filename)
                    shutil.copy2(src_path, dst_path)
                    
                    # Update path in data object for template to use relative path
                    kf["relative_path"] = f"assets/{dst_filename}"
        
        # Render Template
        template = self.env.get_template("document.html")
        html_content = template.render(
            title=project_name, 
            segments=segments,
            video_path=data.get("video_path", ""),
            css_path="assets/style.css"  # Pass relative path to template
        )
        
        # Copy CSS to assets
        css_src = os.path.join(os.path.dirname(__file__), "assets", "style.css")
        if os.path.exists(css_src):
             shutil.copy2(css_src, os.path.join(assets_dir, "style.css"))
        else:
            print("Warning: CSS file not found, skipping copy.")

        output_file = os.path.join(project_dir, "index.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"Document generated at: {output_file}")
        return output_file
