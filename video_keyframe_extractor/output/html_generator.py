from jinja2 import Environment, FileSystemLoader, select_autoescape
import json
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
        
        # Symlink video into assets for same-origin playback
        video_path_abs = data.get("video_path", "")
        video_relative = ""
        if video_path_abs:
            video_path_abs = os.path.abspath(video_path_abs)
            video_ext = os.path.splitext(video_path_abs)[1] or ".mp4"
            video_link = os.path.join(assets_dir, f"video{video_ext}")
            if os.path.islink(video_link):
                os.remove(video_link)
            if os.path.exists(video_path_abs):
                os.symlink(video_path_abs, video_link)
                video_relative = f"assets/video{video_ext}"

        template = self.env.get_template("document.html")
        html_content = template.render(
            title=project_name,
            segments=segments,
            video_path=video_relative,
            cost_info=data.get("cost_info", {}),
            css_path="assets/style.css"
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

        # Save data as JSON for re-rendering without reprocessing
        json_file = os.path.join(project_dir, "data.json")
        json_data = {
            "video_path": video_relative,
            "video_path_abs": video_path_abs,
            "cost_info": data.get("cost_info", {}),
            "segments": [
                {
                    "title": s.get("title", ""),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    "text": s.get("text", ""),
                    "transcript": s.get("transcript", ""),
                    "tasks": s.get("tasks", []),
                    "keyframe": {"relative_path": s["keyframe"]["relative_path"]}
                        if s.get("keyframe") and s["keyframe"].get("relative_path") else None,
                }
                for s in segments
            ],
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"Document generated at: {output_file}")
        return output_file

    def generate_from_cache(self, cached_data: dict, project_name: str) -> str:
        """Re-render HTML from cached data.json without reprocessing."""
        print(f"Regenerating HTML for project: {project_name}")

        project_dir = os.path.join(self.output_dir, project_name)
        assets_dir = os.path.join(project_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Copy CSS
        css_src = os.path.join(os.path.dirname(__file__), "assets", "style.css")
        if os.path.exists(css_src):
            shutil.copy2(css_src, os.path.join(assets_dir, "style.css"))

        # Recreate video symlink if absolute path available
        video_path = cached_data.get("video_path", "")
        video_abs = cached_data.get("video_path_abs", "")
        if video_abs and os.path.exists(video_abs):
            video_ext = os.path.splitext(video_abs)[1] or ".mp4"
            video_link = os.path.join(assets_dir, f"video{video_ext}")
            if os.path.islink(video_link):
                os.remove(video_link)
            os.symlink(video_abs, video_link)
            video_path = f"assets/video{video_ext}"

        template = self.env.get_template("document.html")
        html_content = template.render(
            title=project_name,
            segments=cached_data.get("segments", []),
            video_path=video_path,
            cost_info=cached_data.get("cost_info", {}),
            css_path="assets/style.css",
        )

        output_file = os.path.join(project_dir, "index.html")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Document regenerated at: {output_file}")
        return output_file
