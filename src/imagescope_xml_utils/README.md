# imagescope_xml_utils
 A tool for read and write aperio imagescope xml file.  

Types of tags currently supported for reading and writing: boxes, contours, arrows, ellpises.  
Metadata reads and writes are currently not supported.  

# Attention
### Please note that the coordinate format I use is yx, not xy.
### Color is a int tuple (R, G, B) 

# Dependent
```
lxml
numpy
```

# How to use

This is a simple example.  
Only read a xml file and write it with not modified.  
Your can find the test data in this repo.  

```python
reader = ImageScopeXmlReader("test.xml", keep_arrow_tail=False, use_box_y1x1y2x2=True)
arrows, arrow_colors = reader.get_arrows()
boxes, box_colors = reader.get_boxes()
contours, contour_colors = reader.get_contours()
ellipses, ellipse_colors = reader.get_ellipses()

writer = ImageScopeXmlWriter()
writer.add_arrows(arrows, arrow_colors)
writer.add_boxes(boxes, box_colors)
writer.add_contours(contours, contour_colors)
writer.add_ellipses(ellipses, ellipse_colors)
writer.write('test2.xml')
```

# Format

boxes (use_box_y1x1y2x2=True):  [y1x1y2x2, y1x1y2x2, ...]  
boxes (use_box_y1x1y2x2=False): [[top_left_yx, top_right_yx, bottom_right_yx, bottom_left_yx], [top_left_yx, top_right_yx, bottom_right_yx, bottom_left_yx], ...]  

arrows (keep_arrow_tail=True):  [[head_yx, tail_yx], [head_yx, tail_yx], ...]  
arrows (keep_arrow_tail=False): [head_yx, head_yx, ...]  

contours : [[con_yx, con_yx, ...], [con_yx, con_yx, ...], ...]  

ellipses : [[top_left_yx, bottom_right_yx], [top_left_yx, bottom_right_yx], ...]  
