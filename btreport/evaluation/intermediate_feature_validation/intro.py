import langextract as lx
import textwrap


def main():
    # 1. Define the prompt and extraction rules
    # prompt = textwrap.dedent("""\
    #     Extract characters, emotions, and relationships in order of appearance.
    #     Use exact text for extractions. Do not paraphrase or overlap entities.
    #     Provide meaningful attributes for each entity to add context.""")

    prompt = textwrap.dedent("""\
        Extract characters, emotions, and relationships in order of appearance.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context.""")


    example_text = textwrap.dedent("""\
    "EXAMINATION:  
    MRI BRAIN WO/W CONTRAST
    
    CLINICAL INDICATION:
    new seizure, CT concerning for vasogenic edema/mass, eval for lesion
    
    
    TECHNIQUE:
    MRI Head  without and with contrast: Tumor (Primary) (B 2PT)  
    Non-contrast:  Axial T1, T2, DWI,   Post-contrast : Sagittal 3D FLAIR. Sagittal, axial and coronal T1 
    
    
    CONTRAST:
    GADOTERIDOL 279.3 MG/ML IV SOLN,5 mmol Intravenous,12/10/2023 1944
    
    COMPARISON:
    CT head 12/10/2023.
    
    FINDINGS:
    MASS EFFECT & VENTRICLES: There is mass effect on the left lateral ventricle occipital horn, with slightly prominent left temporal horn compared to the right side. Otherwise the ventricles are normal in size.
    BRAIN:  There is a mass along the posterior medial left parietal occipital lobe with thick irregular peripheral enhancement and central necrosis. This demonstrates intrinsic restricted diffusion along the enhancing portions suggesting high cellularity and measures approximately 2.0 x 1.6 cm in axial dimension (801/93) and 2.7 cm in craniocaudal dimension (702/193). There is surrounding vasogenic edema extending along the paramedian left parietal lobe, the mesial temporal lobe, splenium of the corpus callosum (601/19), as well adjacent left temporoparietal white matter. There is evidence of gyral expansion in the adjacent left paramedian parietal lobe cortex.
    No susceptibility artifact. No additional sites of abnormal enhancement.
    VASCULAR:  Cavernous carotid, vertebral, and other intracranial vascular flow voids are normal
    EXTRA-AXIAL:  There is suggestion of a dural tail adjacent to the mass which abuts the posterior aspect of the falx (702/187). 
    EXTRA-CRANIAL:   Skull and facial bones are normal. Sinuses and mastoids are clear. Orbits are normal. 
    
    IMPRESSION
    1. Left medial occipitoparietal cortically based 2.7 cm cystic and solid enhancing mass with surrounding vasogenic edema. Differential includes high grade glial tumors such as a noncalcified oligodendroglioma given its cortical location and middle age demographics, solitary metastasis, or other rarer cortical tumors such as a pleomorphic xanthoastrocytoma given the well defined nodule, dural tail and relatively young age, although 39 is older than typical of these tumors.
    2. There is near effacement of the left lateral ventricle occipital horn/atria with subtle asymmetric prominence of the left temporal horn. This could represent early entrapped ventricle. Recommend attention on follow-up."
        
    """)


    # 2. Provide a high-quality example to guide the model
    examples = [
        lx.data.ExampleData(
            text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"}
                ),
                lx.data.Extraction(
                    extraction_class="emotion",
                    extraction_text="But soft!",
                    attributes={"feeling": "gentle awe"}
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"}
                ),
            ]
        )
    ]

    input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="llama3:70b",  # Automatically selects Ollama provider
        model_url="http://localhost:11434",
        fence_output=False,
        use_schema_constraints=False
    )

    # Save the results to a JSONL file
    lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

    # Generate the visualization from the file
    html_content = lx.visualize("extraction_results.jsonl")
    with open("visualization.html", "w") as f:
        if hasattr(html_content, 'data'):
            f.write(html_content.data)  # For Jupyter/Colab
        else:
            f.write(html_content)



if __name__ == '__main__':
    main()