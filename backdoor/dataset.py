import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams['font.family'] = 'DejaVu Sans'  # default supports many unicode icons

def box(ax, x, y, w, h, label, icon, fc):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           ec='black', fc=fc, lw=1)
    ax.add_patch(patch)
    # icon top line, label below
    ax.text(x + w/2, y + h*0.62, icon, fontsize=16, ha='center', va='center')
    ax.text(x + w/2, y + h*0.25, label, fontsize=8, ha='center', va='center')
    return (x, y, w, h)

def mid_right(b):
    x,y,w,h = b
    return (x+w, y+h/2)
def mid_left(b):
    x,y,w,h = b
    return (x, y+h/2)
def arrow(ax, p1, p2, color='black', style='-'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='->', lw=1, color=color,
                                 mutation_scale=8, linestyle=style))

fig, ax = plt.subplots(figsize=(13,7))
ax.axis('off')

# colors
enc_col="#C9E7C6"
diff_col="#FFE3B1"
chan_col="#C7DBF4"
bd_col="#F8C8C4"
trig_col="#FFF4B2"
clean_header="#D7CDEB"

row_h=0.12
row_gap=0.25
row1_y=0.70
row2_y=row1_y-row_gap
start_x=0.05
w=0.14
gap=0.018

# Header boxes
box(ax, start_x, row1_y+row_h+0.01, 0.87, 0.05, "Clean Transmission Path", "✔️", clean_header)
box(ax, start_x, row2_y+row_h+0.01, 0.87, 0.05, "Backdoor Attack Path (τ present)", "⚠️", bd_col)

# Clean path boxes
c1=box(ax, start_x, row1_y, w, row_h, "Image $x$", "🖼", trig_col)
c2=box(ax, start_x+(w+gap), row1_y, w, row_h, "Encoder $E_{SD}$", "🗜", enc_col)
c3=box(ax, start_x+2*(w+gap), row1_y, w, row_h, "DDIM Inv.\n$T_{F,1}$", "↩️", diff_col)
c4=box(ax, start_x+3*(w+gap), row1_y, w, row_h, "Channel", "📶", chan_col)
c5=box(ax, start_x+4*(w+gap), row1_y, w, row_h, "Fwd Diff.\n$T_{F,2}$", "🔄", diff_col)
c6=box(ax, start_x+5*(w+gap), row1_y, w, row_h, "DDIM Samp.\nU-Net", "🎯", diff_col)
c7=box(ax, start_x+6*(w+gap), row1_y, w, row_h, "Decoder $D_{SD}$", "🗜", enc_col)
c8=box(ax, start_x+7*(w+gap), row1_y, w, row_h, "Recons.\n$\\hat{x}$", "✅", trig_col)

clean_boxes=[c1,c2,c3,c4,c5,c6,c7,c8]
for b1,b2 in zip(clean_boxes[:-1], clean_boxes[1:]):
    arrow(ax, mid_right(b1), mid_left(b2))
# annotate z_0 and y=z+n
ax.text((mid_right(c1)[0]+mid_left(c2)[0])/2, mid_right(c1)[1]+0.03, r"$z_{0}$", fontsize=9, ha='center')
ax.text((mid_right(c3)[0]+mid_left(c4)[0])/2, mid_right(c3)[1]+0.03, r"$z_{T_{F,1}}$", fontsize=9, ha='center')
ax.text((mid_right(c4)[0]+mid_left(c5)[0])/2, mid_right(c4)[1]+0.03, r"$y=z+n$", fontsize=9, ha='center')


# Attack path boxes
a1=box(ax, start_x, row2_y, w, row_h, "Image $x+\\tau$", "🖼", trig_col)
a2=box(ax, start_x+(w+gap), row2_y, w, row_h, "Encoder $E_{SD}$", "🗜", enc_col)
a3=box(ax, start_x+2*(w+gap), row2_y, w, row_h, "DDIM Inv.\n$T_{F,1}$", "↩️", diff_col)
a4=box(ax, start_x+3*(w+gap), row2_y, w, row_h, "Channel", "📶", chan_col)
a5=box(ax, start_x+4*(w+gap), row2_y, w, row_h, "Fwd Diff.\n$T_{F,2}$", "🔄", diff_col)
a6=box(ax, start_x+5*(w+gap), row2_y, w, row_h, "DDIM Samp.\nU-Net★", "🎯", bd_col)
a7=box(ax, start_x+6*(w+gap), row2_y, w, row_h, "Decoder $D_{SD}$", "🗜", enc_col)
a8=box(ax, start_x+7*(w+gap), row2_y, w, row_h, "Target\n$y^{*}$", "🎯", trig_col)

attack_boxes=[a1,a2,a3,a4,a5,a6,a7,a8]
for b1,b2 in zip(attack_boxes[:-1], attack_boxes[1:]):
    arrow(ax, mid_right(b1), mid_left(b2), color='firebrick')

# Trigger synthesis module
tg=box(ax, start_x-0.03, row2_y-0.18, w+0.04, row_h, "Trigger\nSynthesis", "⚙️", diff_col)
# dashed arrows
arrow(ax, (tg[0]+tg[2]/2, tg[1]+tg[3]), (a1[0]+a1[2]/2, a1[1]), style='--', color='gray')
arrow(ax, (tg[0]+tg[2]/2, tg[1]+tg[3]), (a6[0]+a6[2]/2, a6[1]+a6[3]), style='--', color='gray')

# Legend
legend_patches=[Rectangle((0,0),1,1,fc=enc_col,ec='none',label='Encoder/Decoder'),
                Rectangle((0,0),1,1,fc=diff_col,ec='none',label='Diffusion Step'),
                Rectangle((0,0),1,1,fc=chan_col,ec='none',label='Wireless Channel'),
                Rectangle((0,0),1,1,fc=bd_col,ec='none',label='Backdoored U-Net'),
                Rectangle((0,0),1,1,fc=trig_col,ec='none',label='Trigger / Target')]
ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/data/semcom_backdoor_professional.png', dpi=300, bbox_inches='tight')

