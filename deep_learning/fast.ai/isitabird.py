import fastbook as fb

urls = fb.search_images_ddg('bird photos', max_images=1)
print(len(urls), urls[0])

dest = fb.Path('bird.jpg')
if not dest.exists(): 
    fb.download_url(urls[0], dest, show_progress=False)

im = fb.Image.open(dest)
im.to_thumb(256, 256)

searches = 'forest', 'bird'
path = fb.Path('bird_or_not')

if not path.exists():
    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = fb.search_images_ddg(f'{o} photo')
        fb.download_images(dest, urls=results[:200])
        fb.resize_images(dest, max_size=400, dest=dest)

failed = fb.verify_images(fb.get_images_files(path))
failed.map(fb.Path.unlink)
